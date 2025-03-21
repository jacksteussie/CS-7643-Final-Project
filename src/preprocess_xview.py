# import os
# import json
# import argparse
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from concurrent.futures import ProcessPoolExecutor, as_completed

# from utils import get_image, get_labels, chip_image
# from const import XVIEW_DIR

# def process_image(image_info):
#     """ Function to process a single image for multiprocessing """
#     img_name, image_path, split, output_image_dir, chunk_size, train_labels = image_info
#     chunk_w, chunk_h = chunk_size

#     img = get_image(image_path)

#     if img.shape[-1] == 4:
#         img = img[..., :3]  # Remove alpha channel

#     if split == "train" and train_labels is not None:
#         coords, chips, classes = train_labels
#         img_coords = coords[chips == img_name]
#         img_classes = classes[chips == img_name]
#     else:
#         img_coords = np.zeros((0, 4))  
#         img_classes = np.zeros((0,))

#     # Process image into chips
#     chips_data, chip_boxes, chip_labels = chip_image(img, img_coords, img_classes, shape=(chunk_w, chunk_h))

#     results = []
#     for chip_id, chip in enumerate(chips_data):
#         chip_filename = f"{img_name.replace('.tif', '')}_chip_{chip_id}.jpg"
#         chip_path = os.path.join(output_image_dir, chip_filename)

#         Image.fromarray(chip).convert("RGB").save(chip_path, "JPEG", quality=100)

#         if split == "train":
#             label_filename = chip_filename.replace(".jpg", ".json")
#             label_data = {
#                 "boxes": chip_boxes[chip_id].tolist(),
#                 "labels": chip_labels[chip_id].tolist()
#             }
#             with open(os.path.join(output_image_dir, label_filename), "w") as f:
#                 json.dump(label_data, f)

#         results.append(chip_filename)
    
#     return results

# def process_xview(data_dir, chunk_size=(300, 300), num_workers=4):
#     """ Main function to process xView dataset with multiprocessing """

#     chunk_w, chunk_h = chunk_size
#     output_dir = os.path.join(data_dir, "xview_processed")
#     os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

#     label_file = os.path.join(data_dir, "xView_train.geojson")
#     if os.path.exists(label_file):
#         coords, chips, classes = get_labels(label_file)
#         train_labels = (coords, chips, classes)
#     else:
#         train_labels = None

#     tasks = []
#     for split in ["train", "val"]:
#         image_dir = os.path.join(data_dir, split)
#         output_image_dir = os.path.join(output_dir, split)
#         os.makedirs(output_image_dir, exist_ok=True)

#         for img_name in os.listdir(image_dir):
#             if img_name.endswith(".tif"):
#                 image_path = os.path.join(image_dir, img_name)
#                 tasks.append((img_name, image_path, split, output_image_dir, chunk_size, train_labels if split == "train" else None))

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         future_to_img = {executor.submit(process_image, task): task[0] for task in tasks}

#         for future in tqdm(as_completed(future_to_img), total=len(tasks), desc="Processing images"):
#             future.result()  


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Preprocess xView dataset using multiprocessing for faster image processing.")
#     parser.add_argument("data_dir", type=str, nargs="?", default=XVIEW_DIR, help="Path to xview dataset directory.")
#     parser.add_argument("--chunk_width", type=int, default=300, help="Width of image chunks.")
#     parser.add_argument("--chunk_height", type=int, default=300, help="Height of image chunks.")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for multiprocessing.")

#     args = parser.parse_args()
#     process_xview(args.data_dir, (args.chunk_width, args.chunk_height), args.num_workers)

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split  

from utils import get_image, get_labels
from const import XVIEW_DIR

def chip_image_with_overlap(img, coords, classes, shape=(300, 300), overlap=50):
    """
    Chips an image with a given overlap and gets relative coordinates and classes.
    
    Args:
        img: The input image array (H, W, C).
        coords: (N, 4) array of bounding boxes.
        classes: (N, 1) array of object classes.
        shape: (W, H) tuple for the chip size.
        overlap: Number of pixels for overlapping between adjacent chips.

    Output:
        Chipped images and updated bounding boxes for each chip.
    """
    height, width, _ = img.shape
    wn, hn = shape  # Chip width & height

    # Calculate step size with overlap
    step_x = wn - overlap
    step_y = hn - overlap

    w_num = (width - overlap) // step_x
    h_num = (height - overlap) // step_y

    images = []
    total_boxes = {}
    total_classes = {}

    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x_start, y_start = i * step_x, j * step_y
            x_end, y_end = x_start + wn, y_start + hn

            # Extract chip
            chip = img[y_start:y_end, x_start:x_end, :3]
            images.append(chip)

            # Adjust bounding boxes for this chip
            x_mask = np.logical_and(coords[:, 0] >= x_start, coords[:, 2] <= x_end)
            y_mask = np.logical_and(coords[:, 1] >= y_start, coords[:, 3] <= y_end)
            mask = np.logical_and(x_mask, y_mask)

            valid_boxes = coords[mask]
            valid_classes = classes[mask]

            if valid_boxes.shape[0] > 0:
                # Adjust coordinates to be relative to the chip
                valid_boxes[:, [0, 2]] -= x_start
                valid_boxes[:, [1, 3]] -= y_start
                total_boxes[k] = valid_boxes.tolist()
                total_classes[k] = valid_classes.tolist()
            else:
                total_boxes[k] = [[0, 0, 0, 0]]
                total_classes[k] = [0]

            k += 1

    return images, total_boxes, total_classes


def process_image(image_info):
    """ Function to process a single image for multiprocessing """
    img_name, image_path, split, output_image_dir, chunk_size, overlap, train_labels = image_info
    chunk_w, chunk_h = chunk_size

    img = get_image(image_path)

    # Convert RGBA to RGB (if needed)
    if img.shape[-1] == 4:
        img = img[..., :3]  # Remove alpha channel

    # If in train split, get labels; otherwise, leave them empty
    if split in ["train", "val"] and train_labels is not None:
        coords, chips, classes = train_labels
        img_coords = coords[chips == img_name]
        img_classes = classes[chips == img_name]

        # Ensure img_coords is always 2D
        if img_coords.size == 0:
            img_coords = np.zeros((0, 4))  
            img_classes = np.zeros((0,))
    else:
        # Ensure empty coords/classes for validation images
        img_coords = np.zeros((0, 4))  
        img_classes = np.zeros((0,))

    # Chip the image with overlap
    chips_data, chip_boxes, chip_classes = chip_image_with_overlap(img, img_coords, img_classes, shape=(chunk_w, chunk_h), overlap=overlap)

    results = []
    for chip_id, chip in enumerate(chips_data):
        chip_filename = f"{img_name.replace('.tif', '')}_chip_{chip_id}.jpg"
        chip_path = os.path.join(output_image_dir, chip_filename)

        # Save chip as JPG
        Image.fromarray(chip).convert("RGB").save(chip_path, "JPEG", quality=95)

        if split == "train":
            label_filename = chip_filename.replace(".jpg", ".json")
            label_data = {
                "bboxes": chip_boxes[chip_id],
                "classes": chip_classes[chip_id]
            }
            with open(os.path.join(output_image_dir, label_filename), "w") as f:
                json.dump(label_data, f)

        results.append(chip_filename)
    
    return results


def process_xview(data_dir, chunk_size=(300, 300), overlap=50, train_ratio=0.8, num_workers=4):
    """ Main function to process xView dataset with multiprocessing """

    chunk_w, chunk_h = chunk_size
    output_dir = os.path.join(data_dir, "xview_processed")
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

    # Load labels for train set (if available)
    label_file = os.path.join(data_dir, "xView_train.geojson")
    if os.path.exists(label_file):
        coords, chips, classes = get_labels(label_file)
        train_labels = (coords, chips, classes)
    else:
        train_labels = None

    # Split the dataset into train and validation sets (no data leakage)
    image_dir = os.path.join(data_dir, "train")
    all_images = [img for img in os.listdir(image_dir) if img.endswith(".tif")]

    train_images, val_images = train_test_split(all_images, train_size=train_ratio, random_state=42)

    # Prepare images for multiprocessing
    tasks = []
    for split, images in [("train", train_images), ("val", val_images)]:
        output_image_dir = os.path.join(output_dir, split)
        os.makedirs(output_image_dir, exist_ok=True)

        for img_name in images:
            image_path = os.path.join(image_dir, img_name)
            tasks.append((img_name, image_path, split, output_image_dir, chunk_size, overlap, train_labels))

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_img = {executor.submit(process_image, task): task[0] for task in tasks}

        for future in tqdm(as_completed(future_to_img), total=len(tasks), desc="Processing images"):
            future.result()  # Ensures any errors are raised


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess xView dataset with overlapping image chunks and data leakage prevention.")
    parser.add_argument("data_dir", type=str, nargs="?", default=XVIEW_DIR, help="Path to xview dataset directory.")
    parser.add_argument("--chunk_width", type=int, default=300, help="Width of image chunks.")
    parser.add_argument("--chunk_height", type=int, default=300, help="Height of image chunks.")
    parser.add_argument("--overlap", type=int, default=50, help="Number of pixels for overlapping between chunks.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data (e.g., 0.8 for 80% training, 20% validation).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for multiprocessing.")

    args = parser.parse_args()
    process_xview(args.data_dir, (args.chunk_width, args.chunk_height), args.overlap, args.train_ratio, args.num_workers)
