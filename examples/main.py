import argparse
import os
from typing import List
import cv2
import matplotlib.pyplot as plt
from path import Path
from word_detector import detect, prepare_img, sort_multiline


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp', '*.jpeg']:
        res += Path(data_dir).files(ext)
    return res


def save_word_images(detections, img, output_dir, scale_factor):
    """Save individual word images and create a text file with their details."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'word_labels.txt'), 'w') as f:
        # Sort detections first by vertical position (y) and then by horizontal position (x)
        detections = sorted(detections, key=lambda det: (det.bbox.y, det.bbox.x))

        for idx, det in enumerate(detections):
            # Adjust bounding box coordinates to match the original image size
            x = int(det.bbox.x * scale_factor)
            y = int(det.bbox.y * scale_factor)
            w = int(det.bbox.w * scale_factor)
            h = int(det.bbox.h * scale_factor)

            word_img = img[y:y + h, x:x + w]  # Crop the original image based on the adjusted coordinates
            word_filename = f"word_{idx}.png"
            word_path = os.path.join(output_dir, word_filename)
            cv2.imwrite(word_path, word_img)
            f.write(f"Word {idx}: {word_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.abspath('../data/page'))
    parser.add_argument('--output_dir', type=str, default='./output_words')
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=5)
    parser.add_argument('--theta', type=float, default=5)
    parser.add_argument('--min_area', type=int, default=200)
    parser.add_argument('--img_height', type=int, default=2150)
    parsed = parser.parse_args()

    for fn_img in get_img_files(Path(parsed.data)):
        print(f'Processing file {fn_img}')

        # Load the original image
        img = cv2.imread(fn_img)

        # Get the height of the original image for scaling
        original_height, original_width = img.shape[:2]

        # Resize the image for processing
        processed_img = prepare_img(img.copy(), parsed.img_height)
        processed_height, processed_width = processed_img.shape[:2]

        # Compute the scale factor between the original and processed image
        scale_factor = original_height / processed_height

        # Detect words in the resized image
        detections = detect(processed_img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=parsed.theta,
                            min_area=parsed.min_area)

        if not detections:
            print(f"No words detected in {fn_img}. Try adjusting parameters.")
            continue

        # Save word images from the original image, using scaled bounding box coordinates
        save_word_images(detections, img, parsed.output_dir, scale_factor)

        # Sort detections into lines and plot bounding boxes
        lines = sort_multiline(detections)

        plt.imshow(processed_img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}', color='blue')

        plt.show()


if __name__ == '__main__':
    main()
