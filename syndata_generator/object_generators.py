import glob
import os
import numpy as np
import kubric as kb
import uuid
import logging

logger = logging.getLogger(__name__)


def get_object_files(class_type, tx_asset_directory):
    obj_files = glob.glob(os.path.join(tx_asset_directory, "**/*.obj"), recursive=True)
    
    if class_type == "bottle":
        return [obj_file for obj_file in obj_files if "bottle" in obj_file]
    elif class_type == "can":
        return [obj_file for obj_file in obj_files if "can" in obj_file]


def create_object(class_type, render_filename, position=(0, 0, 0)):
    object_name = str(uuid.uuid4())
    obj = kb.FileBasedObject(
        asset_id=class_type,
        name=object_name,
        position=position,
        render_filename=render_filename,
    )
    obj.quaternion = kb.Quaternion(axis=(1, 0, 0), angle=np.pi / 2)
    return obj


def get_rand_tx_object(
    number_to_get=5, xy_scale=5, class_type="bottle", tx_asset_directory=""
):
    render_filenames = get_object_files(class_type, tx_asset_directory)
    objects = []

    for _ in range(number_to_get):
        render_filename = np.random.choice(render_filenames)
        position = (
            np.random.uniform(-xy_scale, xy_scale),
            np.random.uniform(-xy_scale, xy_scale),
            0,
        )
        obj = create_object(class_type, render_filename, position)
        objects.append(obj)

    return objects


def place_objects_in_row(
    number_to_get=5,
    row_id=0,
    x_spacing=1,
    y_spacing=1,
    class_type="bottle",
    tx_asset_directory="",
):
    render_filenames = get_object_files(class_type, tx_asset_directory)
    objects = []

    start_x = -0.5 * (number_to_get - 1) * x_spacing
    render_filename = np.random.choice(render_filenames)
    logger.info(f"Adding {render_filename} to row {row_id}")

    for i in range(number_to_get):
        position = (start_x + i * x_spacing, row_id * y_spacing, 0)
        obj = create_object(class_type, render_filename, position)
        objects.append(obj)

    return objects
