import kubric as kb
from kubric.renderer import Blender as KubricRenderer
from kubric import file_io
from pathlib import Path
import tensorflow as tf
import numpy as np
import zipfile
import os

HDRI_ASSET = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"
HDRI_DIR = "gs://mv_bckgr_removal/hdri_haven/4k/"

# Location of task zip
BUCKET_NAME = "tx-hyu-task"
ZIP_FILE_PATH = "drink_detection_assigment.zip"


def download_and_unzip_gcs_zip(local_directory):
    # Check if the local directory exists, and create it if it doesn't
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Check if the ZIP file already exists locally
    local_zip_path = os.path.join(local_directory, ZIP_FILE_PATH.lower())
    if os.path.exists(local_zip_path):
        # If the ZIP file exists locally, return
        return

    # Download the ZIP file from GCS if it doesn't exist locally
    with tf.io.gfile.GFile(f"gs://{BUCKET_NAME}/{ZIP_FILE_PATH}", "rb") as remote_file:
        with open(local_zip_path, "wb") as local_file:
            local_file.write(remote_file.read())

    # Unzip the downloaded ZIP file
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(local_directory)


def get_random_hdri():
    """Get a random HDRI from the Kubric public assets."""
    hdri_source = kb.AssetSource.from_manifest(HDRI_ASSET)
    hdri_asset_ids = hdri_source.get_test_split(0.0)[0]  # <- get all train
    hdri_asset_id = np.random.choice(hdri_asset_ids)

    background_hdri = hdri_source.create(asset_id=hdri_asset_id)

    return background_hdri


hdri_source = kb.AssetSource.from_manifest(HDRI_ASSET)


def add_hdri_dome(scene, background_hdri=None):
    """Adding HDRI dome."""

    # Download dome.blend
    dome_path = Path("/kubric/dome.blend")
    if not dome_path.exists():
        tf.io.gfile.copy(HDRI_DIR + "dome.blend", dome_path)

    dome = kb.FileBasedObject(
        name="BackgroundDome",
        position=(0, 0, 0),
        static=True,
        background=True,
        simulation_filename=None,
        render_filename=str(dome_path),
        render_import_kwargs={
            "filepath": str(Path(dome_path) / "Object" / "Dome"),
            "directory": str(Path(dome_path) / "Object"),
            "filename": "Dome",
        },
    )
    scene.add(dome)
    # pylint: disable=import-outside-toplevel
    from kubric.renderer import Blender
    import bpy

    blender_renderer = [v for v in scene.views if isinstance(v, Blender)]
    if blender_renderer:
        dome_blender = dome.linked_objects[blender_renderer[0]]
        if bpy.app.version > (3, 0, 0):
            dome_blender.visible_shadow = False
        else:
            dome_blender.cycles_visibility.shadow = False
        if background_hdri is not None:
            dome_mat = dome_blender.data.materials[0]
            texture_node = dome_mat.node_tree.nodes["Image Texture"]
            texture_node.image = bpy.data.images.load(background_hdri.filename)
    return dome


def sample_point_in_half_sphere_shell(
    inner_radius: float,
    outer_radius: float,
    obj_height: float,
    rng: np.random.RandomState,
):
    """Uniformly sample points that are in a given distance
    range from the origin and with z >= 0."""

    while True:
        v = rng.uniform(
            (-outer_radius, -outer_radius, obj_height / 1.2),
            (outer_radius, outer_radius, obj_height),
        )
        len_v = np.linalg.norm(v)
        correct_angle = True

        if inner_radius <= len_v <= outer_radius and correct_angle:
            return tuple(v)
