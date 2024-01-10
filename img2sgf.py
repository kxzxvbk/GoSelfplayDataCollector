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


def get_estimated_center(orig_est_x: float, orig_est_y: float, points: List, radius: float, momentum=0.99):
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
    res_x = momentum * orig_est_x + (1 - momentum) * (est_x / total_distance)
    res_y = momentum * orig_est_y + (1 - momentum) * (est_y / total_distance)
    return int(res_x), int(res_y)


def estimate_radius(circles: List) -> float:
    """
        Estimate the radius of stones using the detected circles.
    """
    rounded_circles = np.uint16(np.around(circles))
    rounded_circles = rounded_circles.tolist()
    rounded_circles.sort(key=lambda x: x[2])
    radius_tot = [cir[2] for cir in rounded_circles]
    radius = max(radius_tot, key=radius_tot.count)

    recomputed_radius = []
    for cir in circles:
        if abs(cir[2] - radius) <= 2:
            recomputed_radius.append(cir[2])
    radius = sum(recomputed_radius) / len(recomputed_radius)
    return radius


def patchify(checkerboard: np.ndarray) -> List:
    """
        Patchify the board image. Return a list of small image patches that separately contains a crossing of the board.
    """
    gray = cv2.cvtColor(checkerboard, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    max_radius = 30  # TODO: improvements?

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

    # Estimate radius using the detected circles.
    est_radius = estimate_radius(circles)

    # Search an optimal estimated board width using the centers of the detected circles.
    min_board_width = 2 * est_radius - 5
    max_board_width = 2 * est_radius + 5
    x_min = min([circles[i][1] for i in range(len(circles))])
    y_min = min([circles[i][0] for i in range(len(circles))])

    best_board_width = -1
    best_error = 100000
    num_candidates = 1000
    bw = min_board_width
    interval = (max_board_width - min_board_width) / num_candidates
    while bw < max_board_width:
        cur_error = 0
        for cir in circles:
            cur_error += get_nearest(x_min, cir[1], bw)
            cur_error += get_nearest(y_min, cir[0], bw)
        if cur_error < best_error:
            best_error = cur_error
            best_board_width = bw
        bw += interval

    # Calculate a better starting point using all circles.
    estimated_points = []
    started_points = []
    for cir in circles:
        x_tmp, y_tmp = cir[1], cir[0]
        if abs(est_radius - cir[2]) < 2:
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
            if x_cent >= stone.shape[0] + 3 or y_cent >= stone.shape[1] + 3:
                continue
            radius = int(best_board_width / 2)
            cv2.circle(stone, (y_cent, x_cent), radius, (0, 0, 255), 3)

            x_left, x_right = max(0, x_cent - radius), min(stone.shape[0], x_cent + radius)
            y_left, y_right = max(0, y_cent - radius), min(stone.shape[1], y_cent + radius)
            patches.append(checkerboard[x_left: x_right, y_left: y_right, :])

    for cir in circles:
        cv2.circle(stone, (int(cir[0]), int(cir[1])), int(cir[2]), (0, 255, 0), 3)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(stone)
    plt.show()

    return patches


def load_img(path: str):
    src = cv2.imread(path)
    src = cut_img(src)
    return src


if __name__ == '__main__':
    image = load_img("./data/1_board.png")
    segmented_patches = patchify(image)
