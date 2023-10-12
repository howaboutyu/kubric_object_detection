import bpy
import numpy as np

import kubric as kb

import logging


from utils import (
    get_random_hdri,
    add_hdri_dome,
    sample_point_in_half_sphere_shell,
    download_and_unzip_gcs_zip,
    create_walls,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

rng = np.random.default_rng()


def get_random_gray_color():
    v = rng.uniform(0.2, 0.8)
    return kb.Color(r=v, g=v, b=v, a=1.0)


def add_metallic_dome(scene):
    logger.info("Adding metallic dome")

    wall_color = get_random_gray_color()
    wall_material = kb.PrincipledBSDFMaterial(
        color=wall_color,
        metallic=rng.uniform(0.3, 0.8),
        roughness=rng.uniform(0.1, 0.3),
        ior=rng.uniform(2.3, 2.8),
    )

    walls = create_walls(wall_distance=13, material=wall_material)
    scene += walls


def add_hdri_background(scene):
    logger.info("Adding hdri dome")

    background_hdri = get_random_hdri()
    dome = add_hdri_dome(scene, background_hdri=background_hdri)
    # scene += dome

    logger.info(f"Added hdri dome {background_hdri.filename}")


def add_floor(scene):
    floor_color = get_random_gray_color()
    floor_material = kb.PrincipledBSDFMaterial(
        color=floor_color,
        metallic=rng.uniform(0.3, 0.8),
        roughness=rng.uniform(0.1, 0.3),
        ior=rng.uniform(2.3, 2.8),
    )

    floor = kb.Cube(
        name="floor",
        scale=(13.2, 13.2, 0.02),
        position=(0, 0, -0.02),
        material=floor_material,
    )
    scene += floor


def add_random_background(scene):
    if rng.random() > 0.5:
        add_metallic_dome(scene)
    else:
        add_hdri_background(scene)

    if rng.random() > 0.5:
        add_floor(scene)


def random_position_within_bounds(x_bound=(-5, 5), y_bound=(-5, 5), z_bound=(0, 3.5)):
    """Return a random position within the provided bounds."""
    x = rng.uniform(*x_bound)
    y = rng.uniform(*y_bound)
    z = rng.uniform(*z_bound)
    return x, y, z


def indoor_light_color():
    """Simulates typical indoor light color."""
    # Soft white light with a slight variation
    r = rng.uniform(0.9, 1.0)
    g = rng.uniform(0.85, 0.95)
    b = rng.uniform(0.7, 0.8)
    return kb.Color(r, g, b, 1.0)


def random_point_light(name="point_light"):
    position = random_position_within_bounds()
    rand_intensity = rng.uniform(100, 2000)
    color = indoor_light_color()
    return kb.PointLight(
        name=name, position=position, color=color, intensity=rand_intensity
    )


def random_rect_area_light(name="rect_light"):
    position = random_position_within_bounds()
    color = indoor_light_color()
    return kb.RectAreaLight(
        name=name, color=color, intensity=100, width=0.5, height=0.5, position=position
    )


def random_directional_light(name="dir_light"):
    position = random_position_within_bounds()
    color = indoor_light_color()
    rand_look_at = (
        rng.uniform(-1, 1),
        rng.uniform(-1, 1),
        rng.uniform(0, 1),
    )
    rand_intensity = rng.uniform(1.5, 10.5)
    return kb.DirectionalLight(
        name=name,
        position=position,
        look_at=rand_look_at,
        intensity=rand_intensity,
        color=color,
    )


def get_random_lights(num_lights=5):
    light_creators = [
        random_point_light,
        random_rect_area_light,
        random_directional_light,
    ]
    lights = []

    for i in range(num_lights):
        light_creator = rng.choice(light_creators)
        light = light_creator(name=f"{light_creator.__name__}_{i}")
        lights.append(light)

    return lights
