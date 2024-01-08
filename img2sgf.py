import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def cut_img(img: np.ndarray, margin: int = 2):
    """
        Cut off the white margin of the original image. ``margin`` refers to the size of white margin that is not cut.
    """
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_new = cv2.threshold(img_new, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    edges_y, edges_x = np.where(img_new == 0)
    bottom = max(min(edges_y) - margin, 0)
    top = max(edges_y)
    left = max(min(edges_x) - margin, 0)
    right = max(edges_x)
    height = top - bottom + 2 * margin
    width = right - left + 2 * margin
    res_image = img[bottom:bottom + height, left:left + width]

    return res_image


def get_nearest(from_px: float, to_px: float, interval: float):
    """
        Assume ``to_px`` is close to ``from_px`` + ``i`` * ``interval``. Calculate the closest ``i`` * ``interval`` for
    all integer ``i``. Note that the distance here is measured by Euclidian distance.
    """
    nearest_dist = abs(from_px - to_px)
    res = from_px

    while from_px >= to_px:
        from_px -= interval
        if abs(from_px - to_px) < nearest_dist:
            nearest_dist = (from_px - to_px)**2
            res = from_px

    while from_px <= to_px:
        from_px += interval
        if abs(from_px - to_px) < nearest_dist:
            nearest_dist = (from_px - to_px)**2
            res = from_px

    return res


def get_estimated_center(orig_est_x: float, orig_est_y: float, points: List, radius: float):
    """
        Given the original estimated x and y, get a refined coordination which is aligned to the centers of
    detected circles.
    """
    total_distance = 0
    est_x, est_y = 0, 0
    for p in points:
        px, py = p
        dist = 1 / (abs(px - orig_est_x) + abs(py - orig_est_y))
        est_x += dist * get_nearest(from_px=px, to_px=orig_est_x, interval=2 * radius)
        est_y += dist * get_nearest(from_px=py, to_px=orig_est_y, interval=2 * radius)
        total_distance += dist
    return int(est_x / total_distance), int(est_y / total_distance)


def patchify(checkerboard: np.ndarray) -> List:
    """
        Patchify the board image. Return a list of small image patches that separately contains a crossing of the board.
    """
    gray = cv2.cvtColor(checkerboard, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    max_radius = min(gauss.shape) // 19

    # Detect circles in the input image.
    circles = cv2.HoughCircles(
        image=gauss,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=100,
        param2=30,
        maxRadius=max_radius
    )[0]

    # Search an optimal estimated board width using the centers of the detected circles.
    min_board_width = min(checkerboard.shape[:2]) / 20
    max_board_width = min(checkerboard.shape[:2]) / 18
    x_min = min([circles[i][0] for i in range(len(circles))])
    y_min = min([circles[i][1] for i in range(len(circles))])

    best_board_width = -1
    best_error = 100000
    num_candidates = 1000
    bw = min_board_width
    interval = (max_board_width - min_board_width) / num_candidates
    while bw < max_board_width:
        cur_error = 0
        for cir in circles:
            cur_error += get_nearest(x_min, cir[0], bw)
            cur_error += get_nearest(y_min, cir[1], bw)
        if cur_error < best_error:
            best_error = cur_error
            best_board_width = bw
        bw += interval

    # Calculate a better starting point using all circles.
    estimated_points = []
    started_points = []
    for cir in circles:
        x_tmp, y_tmp = cir[0], cir[1]
        estimated_points.append([x_tmp, y_tmp])
        while x_tmp - best_board_width > 0:
            x_tmp -= best_board_width

        while y_tmp - best_board_width > 0:
            y_tmp -= best_board_width

        started_points.append([x_tmp, y_tmp])
    start_x = sum([started_points[i][0] for i in range(len(started_points))]) / len(started_points)
    start_y = sum([started_points[i][1] for i in range(len(started_points))]) / len(started_points)

    # Patchify the image and visualize.
    stone = checkerboard.copy()
    patches = []
    for i in range(19):
        for j in range(19):
            # Get the roughly estimated coordination.
            x_cent, y_cent = start_x + i * best_board_width, start_y + j * best_board_width
            # Get the refined coordination.
            x_cent, y_cent = get_estimated_center(x_cent, y_cent, estimated_points, best_board_width / 2)
            radius = int(best_board_width / 2)
            cv2.circle(stone, (x_cent, y_cent), radius, (0, 0, 255), 3)
            patches.append(stone[x_cent - radius:x_cent + radius, y_cent - radius:y_cent + radius])

    for cir in circles:
        cv2.circle(stone, (int(cir[0]), int(cir[1])), int(cir[2] - 5), (0, 255, 0), 3)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(stone)
    plt.show()

    return patches


def load_img(path: str):
    src = cv2.imread(path)
    src = cut_img(src)
    return src


if __name__ == '__main__':
    image = load_img("./data/97_board.png")
    segmented_patches = patchify(image)
