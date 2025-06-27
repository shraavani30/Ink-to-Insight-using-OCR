from collections import defaultdict
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DetectorRes:
    img: np.ndarray
    bbox: BBox


def detect(img: np.ndarray,
           kernel_size: int,
           sigma: float,
           theta: float,
           min_area: int) -> List[DetectorRes]:
    """Scale space technique for word segmentation with improved handling of words and punctuation.

    Args:
        img: A grayscale uint8 image.
        kernel_size: The size of the filter kernel, must be an odd integer.
        sigma: Standard deviation of Gaussian function used for filter kernel.
        theta: Approximated width/height ratio of words.
        min_area: Ignore word candidates smaller than specified area.

    Returns:
        List of DetectorRes instances, each containing the bounding box and the word image.
    """
    assert img.ndim == 2
    assert img.dtype == np.uint8

    # Apply filter kernel
    kernel = _compute_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    img_thres = 255 - cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Append components to result
    res = []
    components = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in components:
        # Skip small word candidates
        if cv2.contourArea(c) < min_area:
            continue
        # Append bounding box and image of word to result list
        x, y, w, h = cv2.boundingRect(c)

        # Increase the height and width of the bounding box
        padding_x = 5  # Adjust this value to increase the width of the bounding box
        padding_y = 10  # Adjust this value to increase the height of the bounding box
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w += 2 * padding_x
        h += padding_y  # Increase the height for better coverage

        crop = img[y:y + h, x:x + w]
        res.append(DetectorRes(crop, BBox(x, y, w, h)))

    return res


def _compute_kernel(kernel_size: int,
                    sigma: float,
                    theta: float) -> np.ndarray:
    """Compute anisotropic filter kernel."""
    assert kernel_size % 2  # must be odd size

    # Create coordinate grid
    half_size = kernel_size // 2
    xs = ys = np.linspace(-half_size, half_size, kernel_size)
    x, y = np.meshgrid(xs, ys)

    # Compute sigma values in x and y direction
    sigma_y = sigma
    sigma_x = sigma_y * theta

    # Compute terms and combine them
    exp_term = np.exp(-x ** 2 / (2 * sigma_x ** 2) - y ** 2 / (2 * sigma_y ** 2))
    x_term = (x ** 2 - sigma_x ** 2) / (2 * np.pi * sigma_x ** 5 * sigma_y)
    y_term = (y ** 2 - sigma_y ** 2) / (2 * np.pi * sigma_y ** 5 * sigma_x)
    kernel = (x_term + y_term) * exp_term

    # Normalize and return kernel
    kernel = kernel / np.sum(kernel)
    return kernel


def prepare_img(img: np.ndarray, height: int) -> np.ndarray:
    """Convert image to grayscale image (if needed) and resize to given height."""
    assert img.ndim in (2, 3)
    assert height > 0
    assert img.dtype == np.uint8
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def sort_multiline(detections: List[DetectorRes], max_dist: float = 0.7, min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    """Cluster detections into lines, then sort the lines according to x-coordinates of word centers."""
    lines = _cluster_lines(detections, max_dist, min_words_per_line)
    res = []
    for line in lines:
        # Sort words in the line based on their x-coordinates
        line_sorted = sorted(line, key=lambda det: det.bbox.x)
        res.append(line_sorted)
    return res


def _cluster_lines(detections: List[DetectorRes], max_dist: float, min_words_per_line: int) -> List[List[DetectorRes]]:
    """Cluster detections using DBSCAN based on their y-coordinates."""
    y_coords = np.array([det.bbox.y + det.bbox.h / 2 for det in detections]).reshape(-1, 1)
    clustering = DBSCAN(eps=max_dist, min_samples=min_words_per_line).fit(y_coords)

    # Group detections by cluster
    lines = defaultdict(list)
    for label, det in zip(clustering.labels_, detections):
        if label != -1:  # Ignore noise
            lines[label].append(det)

    return list(lines.values())
