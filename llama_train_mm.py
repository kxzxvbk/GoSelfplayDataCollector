import pickle
import random
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from einops.layers.torch import Rearrange

import evaluate
from transformers import TrainingArguments, Trainer, LlamaTokenizer, LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_int8_training

random.seed(0)


def preprocess_data(tokenizer, prompt, img, transform):
    # Prompt before the image.
    pre_prompt = tokenizer.encode(
        'Here is a board of Go game: <img>',
        return_tensors="pt",
        add_special_tokens=True  # bos token should be added.
    )
    # Prompt after the image.
    post_prompt = tokenizer.encode(
        '<\\img> Here is the corresponding explanation: ',
        return_tensors='pt',
        add_special_tokens=False  # bos token should not be added.
    )

    expl = tokenizer.encode(
        prompt + tokenizer.eos_token,
        return_tensors='pt',
        add_special_tokens=False  # bos token should not be added.
    )

    visual_preprocessed = transform(img)

    return {
        'image': visual_preprocessed,
        'pre_prompt': pre_prompt,
        'post_prompt': post_prompt,
        'expl': expl
    }


class GoMMDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_data = self.data[index]
        img, sentence = item_data['board'], item_data['expl']

        # Tokenize and embed mm data.
        inputs = preprocess_data(self.tokenizer, prompt=sentence,
                                 img=img, transform=self.transform)

        return inputs


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            outputs = model(inputs)
            return outputs.loss, outputs

        outputs = model(inputs)
        return outputs.loss


def compute_metrics(pred):
    rouge = evaluate.load('rouge')

    # Convert token ids into string.
    labels_ids = pred.label_ids[..., 1:]
    pred_ids = pred.predictions[0][..., :-1]
    for id, pred in enumerate(pred_ids):
        pred_ids[id][labels_ids[id] == -100] = 2
        pred_ids[id][pred_ids[id] == -100] = 2
        labels_ids[id][labels_ids[id] == -100] = 2
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Calculate metrics.
    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )

    acc_count = 0
    for pred, label in zip(pred_str, label_str):
        if pred == label:
            acc_count += 1

    res_dict = {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
        "acc": round(acc_count / len(label_str), 4)
    }

    return res_dict


def preprocess_logits_for_metrics(logits, labels):
    logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def prepare_dataset(tokenizer):
    with open(f'dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    # Split dataset.
    train_data = data[:-int(len(data) / 5)]
    test_data = data[-int(len(data) / 5):]

    return GoMMDataset(train_data, tokenizer), GoMMDataset(test_data, tokenizer)


class VisionLanguageModel(nn.Module):
    def __init__(self, language_model, lm_tokenizer, vision_model=None, patch_size=16, img_size=224):
        super(VisionLanguageModel, self).__init__()
        self.language_model = language_model
        self.tokenizer = lm_tokenizer
        self.patch_size = patch_size
        self.img_size = img_size

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_size * patch_size, self.language_model.config.hidden_size)
            )

    def _collate_fn(self, batch):
        # Collate data in the same batch.
        bsz = len(batch)
        x, y = [], []
        for i in range(bsz):
            x.append(batch[i]['inputs_embeds'])
            y.append(batch[i]['labels'])
        return {'inputs_embeds': torch.stack(x, dim=0), 'labels': torch.stack(y, dim=0)}

    def forward(self, inputs):
        # (B, T, 4096)
        if isinstance(inputs, Dict):
            inputs = inputs['inputs']
        bsz = len(inputs)
        vision_embeds = self.vision_model(torch.stack([inputs[i]['image'] for i in range(bsz)]))
        vision_t = vision_embeds.shape[1]
        seq_lens = [inputs[i]['pre_prompt'].shape[1] + inputs[i]['post_prompt'].shape[1]
                    + inputs[i]['expl'].shape[1] + vision_t for i in range(bsz)]
        max_len = max(seq_lens)

        embedded_inputs = []
        for i in range(bsz):
            # Pad the sequence into the same length.
            # Left-padding is used.
            if seq_lens[i] < max_len:
                paddings = inputs[i]['pre_prompt'].new_full(
                    (1, max_len - seq_lens[i]), 0
                )
                inputs[i]['pre_prompt'] = torch.cat([paddings, inputs[i]['pre_prompt']], dim=1)

            # language_model: peft model; language_model.model: LlamaForCausalLM; language_model.model: LlamaModel
            pre_embed = self.language_model.model.model.embed_tokens(inputs[i]['pre_prompt']).squeeze(0)
            post_embed = self.language_model.model.model.embed_tokens(inputs[i]['post_prompt']).squeeze(0)
            expl_embed = self.language_model.model.model.embed_tokens(inputs[i]['expl']).squeeze(0)

            # -100 means that this index is ignored when training.
            label = inputs[i]['expl'].new_full(
                (pre_embed.shape[0] + vision_embeds[i].shape[0] + post_embed.shape[0],), -100
            )
            embedded_inputs.append({
                'inputs_embeds': torch.cat([pre_embed, vision_embeds[i], post_embed, expl_embed], dim=0),
                'labels': torch.cat([label, inputs[i]['expl'].squeeze(0)], dim=0)
            })

        embedded_inputs = self._collate_fn(embedded_inputs)
        return self.language_model(**embedded_inputs)


if __name__ == '__main__':
    # Initialize base model.
    base_model = '/mnt/nfs/whl/LLM/llama-2-7b-hf'
    MODE = 'mm'

    # Initialize base model.
    tokenizer = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=True)
    llama_model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, revision='main',
                                                   device_map='auto', load_in_8bit=True)

    # Initialize lora model.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
    )
    llama_model = prepare_model_for_int8_training(llama_model)
    llama_model = get_peft_model(llama_model, peft_config)

    # Initialize vision-language model.
    model = VisionLanguageModel(language_model=llama_model, lm_tokenizer=tokenizer).cuda()
    print(model)
    llama_model.print_trainable_parameters()

    # Prepare dataset.
    train_dataset, test_dataset = prepare_dataset(tokenizer)

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=f'./output_{MODE}',
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=None,
        remove_unused_columns=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        group_by_length=False,
        dataloader_pin_memory=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        tf32=True
    )

    # Trainer
    trainer = ModifiedTrainer(
        model=model,
        data_collator=lambda x: {'inputs': x},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
