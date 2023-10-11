import os
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.core import color
import bpy

import logging

from object_generators import get_rand_tx_object, place_objects_in_row
from environment_generators import get_random_lights, add_random_background
from texture_generators import * 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

rng = np.random.RandomState()
bpy_scene = bpy.context.scene


def setup_scene(resolution, frame_start, frame_end):
    scene = kb.Scene(
        resolution=resolution, frame_start=frame_start, frame_end=frame_end
    )

    return scene


def setup_renderer(scene):
    return KubricRenderer(
        scene, use_denoising=False, adaptive_sampling=True, background_transparency=True
    )


def setup_objects_in_scene(
    scene, num_cans, num_bottles, xy_scale, tx_asset_directory, generation_type
):
    logger.info("Adding objects to scene")
    objects_to_add = []
    if generation_type == "random":
        bottles = get_rand_tx_object(
            number_to_get=num_bottles,
            xy_scale=xy_scale,
            class_type="bottle",
            tx_asset_directory=tx_asset_directory,
        )
        cans = get_rand_tx_object(
            number_to_get=num_cans,
            xy_scale=xy_scale,
            class_type="can",
            tx_asset_directory=tx_asset_directory,
        )
        objects_to_add = bottles + cans
    elif generation_type == "organized":
        num_rows = 2
        row_objects = []
        for row_id in range(num_rows):
            object_to_get = rng.choice(["none", "bottle", "can"], p=[0.1, 0.4, 0.5])
            logger.info(f"Generating row {row_id} with {object_to_get}")
            if object_to_get == "none":
                continue
            row_object = place_objects_in_row(
                number_to_get=rng.randint(2, 9),
                row_id=row_id,
                class_type=object_to_get,
                tx_asset_directory=tx_asset_directory,
            )
            row_objects += row_object
        objects_to_add = row_objects
    scene += objects_to_add


def setup_lights_in_scene(scene):
    logger.info('Adding lights to scene')
    lights = get_random_lights(num_lights=rng.randint(2, 7))
    scene += lights
    if rng.uniform() > 0.5:
        scene += kb.assets.utils.get_clevr_lights(rng=rng)


# def adjust_material_properties(texture_dir=None):
#     for mat in bpy.data.materials:
#         mat_name = mat.name
#         if "transparent" in mat_name or "liquid" in mat_name:
#             # by default the transmission is 0.0
#             mat.node_tree.nodes["Principled BSDF"].inputs[
#                 "Transmission"
#             ].default_value = 1.0

#             #elif "liquid" in mat_name or "label" in mat_name or "cap" in mat_name:
#         elif  "label" in mat_name:
#             logger.info(f'Adjusting material properties for {mat_name}')
#             #add_random_voronoi(mat)
#             #add_random_musgrave(mat)
#             #add_random_perlin_noise(mat)
#             add_image_texture(mat, image_path="/kubric/syndata_generator/output/rgba_00001.png")
        

def adjust_material_properties(texture_dir=None):
    if not texture_dir:
        logger.error("Texture directory not specified!")
        return

    # Get all files in the specified directory
    all_files = [f for f in os.listdir(texture_dir) if os.path.isfile(os.path.join(texture_dir, f))]
    
    # Filter only image files (assuming PNG and JPG formats)
    image_files = [f for f in all_files if f.endswith(('.png', '.jpg'))]
    
    if not image_files:
        logger.error("No image files found in the specified directory!")
        return

    for mat in bpy.data.materials:
        mat_name = mat.name
        if "transparent" in mat_name or "liquid" in mat_name:
            mat.node_tree.nodes["Principled BSDF"].inputs["Transmission"].default_value = 1.0
        elif "label" in mat_name:
            logger.info(f'Adjusting material properties for {mat_name}')
            
            # Randomly select an image from the folder
            selected_image = rng.choice(image_files)
            selected_image_path = os.path.join(texture_dir, selected_image)
            
            # Add the image texture to the material
            add_image_texture(mat, image_path=selected_image_path)

def setup_camera(scene):

    # Randomly set focal_length and sensor_width
    random_focal_length = rng.uniform(40, 60)
    random_sensor_width = rng.uniform(30, 40)

    original_camera_position = (
        rng.uniform(4, 7),
        rng.uniform(4, 7),
        rng.uniform(2, 4),
    )

    scene += kb.PerspectiveCamera(
        name="camera",
        position=original_camera_position,
        focal_length=random_focal_length,
        sensor_width=random_sensor_width,
    )


    # Circular motion
    r = np.sqrt(sum(a * a for a in original_camera_position))
    phi = np.arccos(original_camera_position[2] / r)
    theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))
    theta_change = (2 * np.pi) / (scene.frame_end - scene.frame_start)

    for frame in range(scene.frame_start, scene.frame_end + 1):
        theta_new = theta + frame * theta_change

        x = r * np.cos(theta_new) * np.sin(phi)
        y = r * np.sin(theta_new) * np.sin(phi)
        z = original_camera_position[2] + rng.uniform(-0.2, 0.2)

        scene.camera.position = (x, y, z)
   
        # Common code for look_at and keyframes
        rand_z_look_at = rng.uniform(0, 2)
        scene.camera.look_at((0, 0, rand_z_look_at))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)


def render_and_save_data(renderer, scene, output_dir):


    renderer.save_state(f"{output_dir}/keyframing.blend")

    logger.info("Rendering")
    data_stack = renderer.render()
    logger.info("Done rendering")

    # --- save output files
    output_dir = kb.as_path(output_dir)

    # write rgb
    kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)

    # write metadata including bounding boxes
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [
        asset
        for asset in scene.foreground_assets
        if np.max(asset.metadata["visibility"]) > 0
    ]

    kb.post_processing.compute_bboxes(
        data_stack["segmentation"], visible_foreground_assets
    )

    kb.write_json(
        filename=output_dir / "metadata.json",
        data={"instances": kb.get_instance_info(scene, visible_foreground_assets)},
    )


def generate_synthetic(
    resolution=(300, 300),
    frame_start=1,
    frame_end=3,
    output_dir="output",
    num_cans=1,
    num_bottles=1,
    xy_scale=5,
    tx_asset_directory="local/drink_detection_assigment",
    generation_type="organized",
    texture_dir='generated_textures',
):
    logger.info(f"Generating synthetic data in {output_dir}")
    scene = setup_scene(resolution, frame_start, frame_end)
    renderer = setup_renderer(scene)
    
    setup_objects_in_scene(scene, num_cans, num_bottles, xy_scale, tx_asset_directory, generation_type)
    setup_lights_in_scene(scene)
    adjust_material_properties(texture_dir=texture_dir)
    setup_camera(scene)
    add_random_background(scene)
    render_and_save_data(renderer, scene, output_dir)


if __name__ == "__main__":
    generate_synthetic()
