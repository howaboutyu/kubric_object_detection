""" 
Create tfrecords based on kubric segmentation output 



Expected input:

INPUT_DIR:
    kubric_1/
        rgba_00000.png
        ...
        metadata.json 


OUTPUT_DIR:
    kubric_session_1.tfrecord
    kubric_session_2.tfrecord
"""

import glob
import os
import numpy as np
import tensorflow as tf
import PIL
import io
import json
import hashlib
from collections import defaultdict
import logging
import argparse
import pathlib


import tfrecord_utils as dataset_util


logging.basicConfig(level="INFO")


def read_metadata(json_path):
    """
    Format of metadata.json file - metadata in each kubric simulaton session is stored in a json file.

    Example of a metadata.json file:
        data['instances'][1]['bbox_frames']
            -> [0, 1, 2, 3, 4]
        data['instances'][1]['asset_id']
            -> 'bottle'

        data['instances'][1]['bboxes']
            [
                [0.07, 0.4033333333333333, 0.4166666666666667, 0.5133333333333333],
                ...
            ]

    The output will be a nested dictionary structure with 'frame' as the outer key.
    In each inner dictionary, the key represents the class id (cans or bottles),
    and the corresponding value is a list of bounding boxes.
    The bounding boxes have the format [y_min, x_min, y_max, x_max] in normalized coordinates.

    """

    with tf.io.gfile.GFile(json_path, "rb") as f:
        data = json.load(f)

    instances = data["instances"]

    nested_dict = defaultdict(lambda: defaultdict(list))

    for instance in instances:
        if "asset_id" not in instance:
            continue

        asset_id = instance["asset_id"]
        for frame, bbox in zip(instance["bbox_frames"], instance["bboxes"]):
            nested_dict[frame][asset_id].append(bbox)

    return nested_dict


def frame_to_example(image_path, frame_dict):
    """
    Creates a tfrecord example for a particular frame.
    Args:
        image_path: path to the rgba image
        frame_dict: dictionary for a particular frame with the format described in read_metadata
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} does not exist")

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = PIL.Image.open(encoded_image_io)
    width, height = image.size
    image_key = hashlib.sha256(encoded_image).hexdigest()

    # get bbs and masks
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for asset_id, bbs in frame_dict.items():
        class_id = 1 if asset_id == "bottle" else 2
        for bb in bbs:
            _y_min, _x_min, _y_max, _x_max = bb
            xmin.append(_x_min)
            ymin.append(_y_min)
            xmax.append(_x_max)
            ymax.append(_y_max)
            classes.append(class_id)
            classes_text.append(asset_id.encode("utf8"))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/key/sha256": dataset_util.bytes_feature(
                    image_key.encode("utf8")
                ),
                "image/encoded": dataset_util.bytes_feature(encoded_image),
                "image/format": dataset_util.bytes_feature("png".encode("utf8")),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmin),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmax),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymin),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymax),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return example


def kubric_session_to_tf_example(kubric_session_dir):
    logging.info(f"Processing {kubric_session_dir}")

    all_files = [f for f in os.listdir(kubric_session_dir) if f.endswith(".png")]
    rgba_files = [f for f in all_files if f.startswith("rgba")]

    json_file = os.path.join(kubric_session_dir, "metadata.json")

    if os.path.exists(json_file) == False:
        logging.error(f"json file {json_file} does not exist")
        return []

    metadata_nested_dict = read_metadata(json_file)

    example_list = []

    for rgba_file in rgba_files:
        frame = rgba_file.split("_")[-1].split(".")[0]
        frame = int(frame)
        image_path = os.path.join(kubric_session_dir, rgba_file)

        if frame not in metadata_nested_dict:
            continue

        frame_dict = metadata_nested_dict[frame]

        example = frame_to_example(image_path, frame_dict)

        example_list.append(example)

    return example_list


def write_to_shards(session_dir_list, tfrecord_output_dir, num_writers, suffix=""):
    if not os.path.exists(tfrecord_output_dir):
        os.makedirs(tfrecord_output_dir)

    tfrecord_files = [
        os.path.join(tfrecord_output_dir, f"shard_{i}_{suffix}.tfrecords")
        for i in range(num_writers)
    ]

    writers = [tf.io.TFRecordWriter(output_file) for output_file in tfrecord_files]

    for session_dir in session_dir_list:
        session_id = os.path.basename(session_dir)
        logging.info(f"Writing session {session_id} to shard ")

        example_list = kubric_session_to_tf_example(session_dir)

        for example in example_list:
            writers[np.random.randint(0, num_writers)].write(
                example.SerializeToString()
            )

    for writer in writers:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the input directory and output folder."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output_organized",
        help="Input directory containing sessions.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="test_shards",
        help="Folder where output shards will be saved.",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.9,
        help="Proportion of training data to total data.",
    )
    args = parser.parse_args()

    logging.info(f"Input directory set to: {args.input_dir}")
    logging.info(f"Output folder set to: {args.output_folder}")
    logging.info(f"Train-validation split ratio: {args.train_val_split}")

    session_dir_list = glob.glob(os.path.join(args.input_dir, "*"))

    num_sessions = len(session_dir_list)

    train_list = session_dir_list[: int(args.train_val_split * num_sessions)]
    val_list = session_dir_list[int(args.train_val_split * num_sessions) :]

    write_to_shards(train_list, args.output_folder, 64, "train")
    write_to_shards(val_list, args.output_folder, 8, "val")
