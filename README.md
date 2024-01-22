# README

本 repo 提供了三个部分的工具：
- 一个简单的围棋棋谱标注工具，提供简易的 GUI 接口，通过截图的形式保存每个“棋盘-解释”对。
- 一个转换文件，将使用上述方法得到的原始数据转化为 pkl 格式
- 使用 llama 模型进行多模态微调的训练代码

## 使用教程

首先，运行 `main.py` :

```shell
python main.py
```

1) 点击窗口中的“截棋盘图”按钮，并选择相关屏幕区域
2) 点击窗口中的“截解释图”按钮，并选择相关屏幕区域
3) 检查两个图是否截取正确，如不正确，可以按照上述两个步骤重新截取；如确认正确，点击“下一个”按钮即可创建下一个

最后创建完毕所有的训练样本之后，使用:

```shell
python convert.py
```

将原始数据转化为 pkl 的数据集格式。

最后，使用：

```shell
python llama_train_mm.py
```

开启使用 llama 的训练过程。

## 常见问题

**为什么可截图的区域没有布满我的屏幕？**

- 可以在 `main.py` 中修改 `SCREEN_DPI` 参数进行修改

**为什么截图的区域和最终显示的不同？**

- 这很可能是因为电脑的显示区域比例并非 125%。
- 在 windows 系统中，可以在桌面点击鼠标右键 -> 查看显示设置，得到电脑的具体显示比例
- 在得到了显示比例中，在 `main.py` 中修改 `SCREEN_RATIO` 参数对应上电脑的设置即可
