import logging
import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.core import color
import glob
import os
import uuid
import multiprocessing


from utils import get_random_hdri, add_hdri_dome, sample_point_in_half_sphere_shell

logging.basicConfig(level="INFO")


def generate_synthetic(
    resolution=(300, 300),
    frame_start=1,
    frame_end=3,
    output_dir="output",
    num_cans=1,
    num_bottles=1,
    xy_scale=5,
    tx_asset_directory="/kubric/drink_detection_assigment",
):
    logging.info(f"Generating synthetic data in {output_dir}")

    scene = kb.Scene(
        resolution=resolution, frame_start=frame_start, frame_end=frame_end
    )
    renderer = KubricRenderer(scene)

    # scene += kb.Sphere(name="floor", scale=1000, position=(0, 0, +1000), background=True)
    # scene += kb.Cube(name="floor", scale=(0.5, 0.7, 1.0), position=(0, 0, 1.1))

    # Use glob to find all .obj files in the directory and its subdirectories
    obj_files = glob.glob(os.path.join(tx_asset_directory, "**/*.obj"), recursive=True)

    bottle_obj_files = [obj_file for obj_file in obj_files if "bottle" in obj_file]
    can_obj_files = [obj_file for obj_file in obj_files if "can" in obj_file]

    def get_rand_tx_object(number_to_get=5, xy_scale=5, class_type="bottle"):
        """get random models from he asset folder"""
        objects = []

        if class_type == "bottle":
            render_filenames = bottle_obj_files
        elif class_type == "can":
            render_filenames = can_obj_files

        for _ in range(number_to_get):
            render_filename = np.random.choice(render_filenames)

            # load object
            object = kb.FileBasedObject(
                asset_id=class_type,
                name=str(uuid.uuid4()),
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

    bottles = get_rand_tx_object(
        number_to_get=num_bottles, xy_scale=xy_scale, class_type="bottle"
    )
    cans = get_rand_tx_object(
        number_to_get=num_cans, xy_scale=xy_scale, class_type="can"
    )

    scene += bottles
    scene += cans

    scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4), look_at=(0, 0, 0))

    # lamp_fill = kb.RectAreaLight(
    #     name="lamp_fill",
    #     color=color.Color.from_hexint(0xC2D0FF),
    #     intensity=100,
    #     width=0.5,
    #     height=0.5,
    #     position=(0, 0, 3.01122),
    # )

    # scene += lamp_fill

    scene += kb.DirectionalLight(
        name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5
    )

    scene += kb.DirectionalLight(
        name="sun2", position=(1, 0.5, 3), look_at=(0, 0, 0), intensity=1.5
    )

    # scene.ambient_illumination = kb.Color(1.05, 0.05, 0.05)

    # print('Adding hdri dome')
    # background_hdri = get_random_hdri()
    # dome = add_hdri_dome(scene, background_hdri=background_hdri)
    # scene += dome
    # print('Done adding hdri dome')

    rng = np.random.RandomState()
    scene += kb.assets.utils.get_clevr_lights(rng=rng)
    material = kb.PrincipledBSDFMaterial(
        color=kb.Color.from_hsv(rng.uniform(), rng.uniform(), 1),
        metallic=1.0,
        roughness=0.2,
        ior=2.5,
    )

    floor = kb.Cube(
        name="floor",
        scale=(13.2, 13.2, 0.02),
        position=(0, 0, -0.02),
        material=material,
    )
    scene += floor

    original_camera_position = (4.48113, -4.50764, 3.34367)
    r = np.sqrt(sum(a * a for a in original_camera_position))

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

        phi = np.random.uniform(np.pi / 4, np.pi/2 + np.pi/8)
        z = r * np.cos(phi)
        z_shift_direction = (i % num_phi_values_per_theta) - 1
        z = z + z_shift_direction * 1.2

        scene.camera.position = (x, y, z)
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    # --- save scene for quick inspection
    renderer.save_state(f"{output_dir}/keyframing.blend")

    # --- render the data
    print("Rendering")
    data_stack = renderer.render()
    print("Done rendering")

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

    

def main():
    num_generation = 200 
    for _ in range(num_generation):
        output_dir = os.path.join("output", str(uuid.uuid4()))
        num_cans = np.random.randint(1, 7)
        num_bottles = np.random.randint(1, 7)
        generate_synthetic(           
            resolution=(320, 320),                              
            frame_start=1,
            frame_end=10, 
            output_dir=output_dir,      
            num_cans=num_cans,             
            num_bottles=num_bottles,             
            xy_scale=3,            
            tx_asset_directory="local/drink_detection_assigment"
        )
   
if __name__ == "__main__":
    main()
