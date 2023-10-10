import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.core import color
from kubric import randomness
import glob
import os
import uuid
import bpy
import multiprocessing
import logging
import sys


from utils import (
    get_random_hdri,
    add_hdri_dome,
    sample_point_in_half_sphere_shell,
    download_and_unzip_gcs_zip,
    create_walls,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info("info")

bpy_scene = bpy.context.scene


def get_random_gray_color():
    v = np.random.uniform(0.2, 0.8)
    return kb.Color(r=v, g=v, b=v, a=1.0)


def get_random_lights(num_lights=5):
    lights = []

    for _ in range(num_lights):
        z = np.random.uniform(0, 3.5)
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        position = (x, y, z)

        light_id = np.random.randint(0, 3)

        if light_id == 0:
            light = kb.PointLight(
                name="point_light_0", position=position, intensity=300.0
            )
        if light_id == 1:
            light = kb.RectAreaLight(
                name="lamp_fill",
                color=color.Color.from_hexint(0xC2D0FF),
                intensity=100,
                width=0.5,
                height=0.5,
                position=position,
            )

        if light_id == 2:
            light = kb.DirectionalLight(
                name="sun", position=position, look_at=(0, 0, 0), intensity=1.5
            )

        lights.append(light)

    return lights


def add_random_background(scene):
    """either add a hdri dome or a gray metalic background"""

    if np.random.random() > 0.1:
        logger.info("Adding metalic dome")
        # --- add floor and walls
        floor_color = get_random_gray_color()  # kb.Color(r=0.5, g=0.5, b=0.5, a=1.0)

        # color_label, random_color =
        wall_color = get_random_gray_color()  # kb.Color(r=0.5, g=0.5, b=0.5, a=1.0)
        wall_material = kb.PrincipledBSDFMaterial(
            color=wall_color,
            metallic=0.5,
            roughness=0.2,
            ior=2.5,
        )

        walls = create_walls(wall_distance=13, material=wall_material)

        scene += walls
    else:
        logger.info("Adding hdri dome")
        background_hdri = get_random_hdri()
        dome = add_hdri_dome(scene, background_hdri=background_hdri)
        scene += dome
        logger.info(f"Added hdri dome {background_hdri.filename}")

    floor_material = kb.PrincipledBSDFMaterial(
        color=floor_color,
        metallic=0.5,
        roughness=0.2,
        ior=2.5,
    )

    floor = kb.Cube(
        name="floor",
        scale=(13.2, 13.2, 0.02),
        position=(0, 0, -0.02),
        material=floor_material,
    )
    scene += floor


def get_rand_tx_object(
    number_to_get=5, xy_scale=5, class_type="bottle", tx_asset_directory=""
):
    """get random models from he asset folder"""

    # TODO: move this glob out of the function
    obj_files = glob.glob(os.path.join(tx_asset_directory, "**/*.obj"), recursive=True)

    bottle_obj_files = [obj_file for obj_file in obj_files if "bottle" in obj_file]
    can_obj_files = [obj_file for obj_file in obj_files if "can" in obj_file]

    objects = []

    if class_type == "bottle":
        render_filenames = bottle_obj_files
    elif class_type == "can":
        render_filenames = can_obj_files

    for _ in range(number_to_get):
        render_filename = np.random.choice(render_filenames)

        # load object
        object_name = str(uuid.uuid4())
        object = kb.FileBasedObject(
            asset_id=class_type,
            name=object_name,
            position=(0, 0, 0),
            render_filename=render_filename,
        )

        # random the object but keep it upright
        # object.quaternion = kb.Quaternion(axis=(1, 0, 0), angle=np.random.uniform(0, 2*np.pi))

        # rotate object along x-axis
        object.quaternion = kb.Quaternion(axis=(1, 0, 0), angle=np.pi / 2)

        # random pos along x-y plane
        object.position = (
            np.random.uniform(-xy_scale, xy_scale),
            np.random.uniform(-xy_scale, xy_scale),
            0,
        )

        objects.append(object)

    return objects


def place_objects_in_row(
    number_to_get=5,
    row_id=0,
    x_spacing=1,
    y_spacing=1,
    class_type="bottle",
    tx_asset_directory="",
):
    """Place models from the asset folder along a row."""

    # TODO: move this glob out of the function
    obj_files = glob.glob(os.path.join(tx_asset_directory, "**/*.obj"), recursive=True)

    bottle_obj_files = [obj_file for obj_file in obj_files if "bottle" in obj_file]
    can_obj_files = [obj_file for obj_file in obj_files if "can" in obj_file]

    objects = []

    if class_type == "bottle":
        render_filenames = bottle_obj_files
    elif class_type == "can":
        render_filenames = can_obj_files

    # Calculate the starting x position based on the number of objects and x_spacing
    start_x = -0.5 * (number_to_get - 1) * x_spacing
    render_filename = np.random.choice(render_filenames)
    logger.info(f"Adding {render_filename} to row {row_id}")

    for i in range(number_to_get):
        # load object
        object_name = str(uuid.uuid4())
        object = kb.FileBasedObject(
            asset_id=class_type,
            name=object_name,
            position=(0, 0, 0),
            render_filename=render_filename,
        )

        # rotate object along x-axis
        object.quaternion = kb.Quaternion(axis=(1, 0, 0), angle=np.pi / 2)

        # set pos along x-y plane in a row
        object.position = (start_x + i * x_spacing, row_id * y_spacing, 0)

        objects.append(object)

    return objects


def generate_synthetic(
    resolution=(300, 300),
    frame_start=1,
    frame_end=3,
    output_dir="output",
    num_cans=1,
    num_bottles=1,
    xy_scale=5,
    tx_asset_directory="/kubric/drink_detection_assigment",
    generation_type="organized",  # generation type can be random or organized
):
    rng = np.random.RandomState()
    logger.info(f"random seed: {rng.get_state()}")
    logging.info(f"Generating synthetic data in {output_dir}")

    scene = kb.Scene(
        resolution=resolution, frame_start=frame_start, frame_end=frame_end
    )
    renderer = KubricRenderer(
        scene, use_denoising=True, adaptive_sampling=False, background_transparency=True
    )

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
        # scene += bottles
        num_rows = 7
        row_objects = []
        for row_id in range(num_rows):
            object_to_get = np.random.choice(
                ["none", "bottle", "can"], p=[0.1, 0.4, 0.5]
            )

            logging.info(f"generting row {row_id} with {object_to_get}")
            if object_to_get == "none":
                continue

            row_object = place_objects_in_row(
                number_to_get=np.random.randint(2, 9),
                row_id=row_id,
                x_spacing=1,
                y_spacing=1,
                class_type=object_to_get,
            )

            row_objects += row_object

        objects_to_add = row_objects

    scene += objects_to_add
    # make transparent material have transmission of 1
    for mat in bpy.data.materials:
        mat_name = mat.name
        if "transparent" in mat_name or "liquid" in mat_name:
            mat.node_tree.nodes["Principled BSDF"].inputs[
                "Transmission"
            ].default_value = 1.0

        if "liquid" in mat_name or "label" in mat_name or "cap" in mat_name:
            # --- Generate random procedural texture with Blender nodes
            # from: kubric/challenges/texture_structure_nerf/worker.py
            tree = mat.node_tree

            mat_node = tree.nodes["Principled BSDF"]
            ramp_node = tree.nodes.new(type="ShaderNodeValToRGB")
            tex_node = tree.nodes.new(type="ShaderNodeTexNoise")
            scaling_node = tree.nodes.new(type="ShaderNodeMapping")
            rotation_node = tree.nodes.new(type="ShaderNodeMapping")
            vector_node = tree.nodes.new(type="ShaderNodeNewGeometry")

            tree.links.new(
                vector_node.outputs["Position"], rotation_node.inputs["Vector"]
            )
            tree.links.new(
                rotation_node.outputs["Vector"], scaling_node.inputs["Vector"]
            )
            tree.links.new(scaling_node.outputs["Vector"], tex_node.inputs["Vector"])
            tree.links.new(tex_node.outputs["Fac"], ramp_node.inputs["Fac"])
            tree.links.new(ramp_node.outputs["Color"], mat_node.inputs["Base Color"])

            rotation_node.inputs["Rotation"].default_value = (
                rng.uniform() * np.pi,
                rng.uniform() * np.pi,
                rng.uniform() * np.pi,
            )

            fpower = rng.uniform()
            max_log_frequency = 3.0
            min_log_frequency = -3.0
            frequency = 10 ** (
                fpower * (max_log_frequency - min_log_frequency) + min_log_frequency
            )

            scaling_node.inputs["Scale"].default_value = (
                frequency,
                frequency,
                frequency,
            )

            for x in np.linspace(0.0, 1.0, 5)[1:-1]:
                ramp_node.color_ramp.elements.new(x)

            for element in ramp_node.color_ramp.elements:
                element.color = kb.random_hue_color(rng=rng)

    # --- add camera

    # --- add lights
    lights = get_random_lights(num_lights=np.random.randint(2, 7))
    scene += lights

    if rng.uniform() > 0.5:
        scene += kb.assets.utils.get_clevr_lights(rng=rng)

    # --- add background
    add_random_background(scene)

    original_camera_position = (
        np.random.uniform(5, 7),
        np.random.uniform(5, 7),
        np.random.uniform(0, 5),
    )

    scene += kb.PerspectiveCamera(
        name="camera", position=original_camera_position, look_at=(0, 0, 0)
    )
    r = np.sqrt(sum(a * a for a in original_camera_position))
    r += np.random.uniform(-0.5, 2.5)
    phi = np.arccos(original_camera_position[2] / r)
    theta = np.arccos(original_camera_position[0] / (r * np.sin(phi)))
    num_phi_values_per_theta = 1  # < only circular motion
    theta_change = (2 * np.pi) / (
        (scene.frame_end - scene.frame_start) / num_phi_values_per_theta
    )
    for frame in range(scene.frame_start, scene.frame_end + 1):
        i = frame - scene.frame_start
        theta_new = (i // num_phi_values_per_theta) * theta_change + theta

        # These values of (x, y, z) will lie on the same sphere as the original camera.
        x = r * np.cos(theta_new) * np.sin(phi)
        y = r * np.sin(theta_new) * np.sin(phi)

        phi = np.random.uniform(np.pi / 6, np.pi / 2)
        z = r * np.cos(phi)
        z_shift_direction = (i % num_phi_values_per_theta) - 1
        z = z + z_shift_direction * 1.2

        scene.camera.position = (x, y, z)

        rand_z_look_at = np.random.uniform(0, 2)
        scene.camera.look_at((0, 0, rand_z_look_at))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    # --- save scene for quick inspection
    renderer.save_state(f"{output_dir}/keyframing.blend")

    # --- render the data
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
    """
    kb.compute_visibility(data_stack["segmentation"], scene.assets)

    bottles_segmentation = kb.adjust_segmentation_idxs(
        data_stack["segmentation"], scene.assets, bottles
    ).astype(np.uint8)

    cans_segmentation = kb.adjust_segmentation_idxs(
        data_stack["segmentation"], scene.assets, cans
    ).astype(np.uint8)

    # kb.adjust_segmentation_idxs modifies the idxs to be in the range [0, num_objects]
    # add 100 to the cans segmentation idxs
    cans_segmentation[cans_segmentation > 0] += 100


    data_stack["segmentation"] =  bottles_segmentation + cans_segmentation
    data_stack['segmentation'] = data_stack['segmentation'].astype(np.uint8)

    print(f'Unique segmentation idxs: {np.unique(data_stack["segmentation"])}')
    kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)
    """


if __name__ == "__main__":
    # TODO: Add argparse
    print("Starting")
    tx_assignment_dir = "local"
    logger.info(f"Downloading assets to {tx_assignment_dir}")
    download_and_unzip_gcs_zip(tx_assignment_dir)
    logger.info(f"Done downloading assets to {tx_assignment_dir}")

    num_generation = 200
    for _ in range(num_generation):
        output_dir = os.path.join("output", str(uuid.uuid4()))
        num_cans = np.random.randint(1, 7)
        num_bottles = np.random.randint(1, 7)
        random_y_res = np.random.randint(300, 600)
        random_x_res = np.random.randint(random_y_res, 600)  # ensure x_res >= y_res
        resolution = (random_x_res, random_y_res)
        generate_synthetic(
            resolution=resolution,
            frame_start=1,
            frame_end=5,
            output_dir=output_dir,
            num_cans=num_cans,
            num_bottles=num_bottles,
            xy_scale=3,
            tx_asset_directory=f"{tx_assignment_dir}/drink_detection_assigment",
            generation_type="organized",
        )
