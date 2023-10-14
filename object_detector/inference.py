import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import json


def draw_detections(
    image,
    boxes,
    classes,
    scores,
    output_image_file_name,
    show_detection=False,
    linewidth=1,
):
    """
    Visualize Detection function taken from: https://keras.io/examples/vision/retinanet/
    """
    image = np.array(image, dtype=np.uint8)
    plt.figure()  # . #figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = [1, 0, 0] if _cls == "bottle" else [0, 1, 0]
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        # ax.text(
        #     x1,
        #     y1,
        #     text,
        #     bbox={"facecolor": color, "alpha": 0.4},
        #     clip_box=ax.clipbox,
        #     clip_on=True,
        # )

    plt.savefig(output_image_file_name, bbox_inches="tight", pad_inches=0)
    if show_detection:
        plt.show()

    return ax


def read_image(file_name, target_size=None):
    """
    Read image from file and convert to numpy array.
    resize image if target_size is not None
    """
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if target_size is not None:
        # image = tf.image.resize_with_pad(
        #    image, target_height=target_size, target_width=target_size
        # )
        image = tf.image.resize(
            image, (target_size, target_size), preserve_aspect_ratio=True
        )

    image = (image.numpy() * 255).astype(np.uint8)

    return image


def run_inference(
    model,
    image_path,
    score_threshold=0.5,
    output_image_file_name="detection_result.png",
    output_json_file_name="detection_result.json",
    resize_size=None,
):
    print(f"Running inference on {image_path}")
    print(f"Saving detection result to {output_image_file_name}")

    # original image
    original_img = read_image(image_path)
    height, width, _ = original_img.shape
    print(f"height: {height}, width: {width}")

    # resize image for inference
    img_inf = read_image(image_path, resize_size)

    detections = model([img_inf])

    # Initialize a list to store detection results
    detection_results = []

    num_detections = detections["num_detections"][0].numpy().astype(np.int32)

    for index in range(num_detections):
        score = detections["detection_scores"][0][index].numpy()

        if score < score_threshold:
            continue
        multiclass_score = detections["detection_multiclass_scores"][0][index].numpy()

        class_id = np.argmax(multiclass_score)
        if class_id == 0:
            class_name = "bottle"
        elif class_id == 1:
            class_name = "can"

        bbox = detections["detection_boxes"][0][index].numpy()

        y_min = int(bbox[0] * height)
        x_min = int(bbox[1] * width)
        y_max = int(bbox[2] * height)
        x_max = int(bbox[3] * width)

        # Store detection information in a dictionary
        detection_info = {
            "class": class_name,
            "score": float(score),
            "bbox": [x_min, y_min, x_max, y_max],
        }

        detection_results.append(detection_info)

    # Save detection results to JSON
    output_data = {
        "image_path": image_path,
        "detections": detection_results,
    }

    with open(output_json_file_name, "w") as json_file:
        json.dump(output_data, json_file)

    # Draw detections on the image
    draw_detections(
        original_img,
        [info["bbox"] for info in detection_results],
        [info["class"] for info in detection_results],
        [info["score"] for info in detection_results],
        output_image_file_name=output_image_file_name,
        show_detection=False,
    )


def main(args):
    MODEL_DIR = args.model_dir
    TEST_IMAGE_DIR = args.image_dir
    SCORE_THRESHOLD = args.score_threshold
    OUTPUT_DIR = args.output_dir
    RESIZE_SIZE = args.resize_size
    OUTPUT_JSON_FILE_NAME = args.output_json

    model = tf.saved_model.load(MODEL_DIR)

    img_paths = [
        os.path.join(TEST_IMAGE_DIR, img_name)
        for img_name in os.listdir(TEST_IMAGE_DIR)
    ]

    # make output dir if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for img_path in img_paths:
        base_name = os.path.basename(img_path)
        output_image_file_name = os.path.join(OUTPUT_DIR, base_name)
        run_inference(
            model,
            img_path,
            score_threshold=SCORE_THRESHOLD,
            output_image_file_name=output_image_file_name,
            output_json_file_name=OUTPUT_JSON_FILE_NAME,
            resize_size=RESIZE_SIZE,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Script")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_model",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/Users/david/Documents/telexistence_assignment/telexistence_assignment/drink_detection_assigment/target_images",
        help="Path to the directory containing test images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="Path to the directory containing test images",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="output.json",
        help="output json containing detection results",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold for object detection",
    )
    parser.add_argument(
        "--resize_size", type=int, default=512, help="Size to resize images to"
    )

    args = parser.parse_args()
    main(args)

