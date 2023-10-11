import numpy as np
import bpy

def add_random_voronoi(mat, seed=None):
    # Initialize the RNG with the given seed
    rng = np.random.default_rng(seed)

    tree = mat.node_tree

    # Add Voronoi Texture node
    voronoi_node = tree.nodes.new("ShaderNodeTexVoronoi")

    # Position the Voronoi node to the left of the Principled BSDF node for neatness
    mat_node = tree.nodes["Principled BSDF"]
    voronoi_node.location = (mat_node.location.x - 300, mat_node.location.y)

    # Connect the Voronoi Color output to the Base Color of the Principled BSDF
    links = tree.links
    link = links.new(voronoi_node.outputs["Color"], mat_node.inputs["Base Color"])

    # Randomly set the Voronoi settings using RNG
    voronoi_node.inputs["Scale"].default_value = rng.uniform(1.0, 10.0)

    # For the randomness parameter
    voronoi_node.inputs["Randomness"].default_value = rng.uniform(0.0, 1.0)

    # For setting feature and distance
    feature_options = ['F1', 'F2', 'SMOOTH_F1', 'DISTANCE_TO_EDGE', 'N_SPHERE_RADIUS']
    voronoi_node.feature = rng.choice(feature_options)

    distance_options = ['EUCLIDEAN', 'MANHATTAN', 'CHEBYCHEV', 'MINKOWSKI']
    voronoi_node.distance = rng.choice(distance_options)


def add_random_musgrave(mat, seed=None):
    # Initialize the RNG with the given seed
    rng = np.random.default_rng(seed)

    tree = mat.node_tree

    # Add Musgrave Texture node
    musgrave_node = tree.nodes.new("ShaderNodeTexMusgrave")

    # Position the Musgrave node to the left of the Principled BSDF node for neatness
    mat_node = tree.nodes["Principled BSDF"]
    musgrave_node.location = (mat_node.location.x - 300, mat_node.location.y)

    # Connect the Musgrave Fac output to the Base Color of the Principled BSDF
    links = tree.links
    link = links.new(musgrave_node.outputs["Fac"], mat_node.inputs["Base Color"])

    # Randomly set the Musgrave settings using RNG
    musgrave_node.inputs["Scale"].default_value = rng.uniform(0.5, 10.0)
    musgrave_node.inputs["Detail"].default_value = rng.uniform(0.0, 16.0)
    musgrave_node.inputs["Dimension"].default_value = rng.uniform(0.0, 1.0)
    musgrave_node.inputs["Lacunarity"].default_value = rng.uniform(0.0, 6.0)
    musgrave_node.inputs["Offset"].default_value = rng.uniform(-1.0, 1.0)
    musgrave_node.inputs["Gain"].default_value = rng.uniform(0.0, 6.0)

    # For setting type
    type_options =['MULTIFRACTAL', 'RIDGED_MULTIFRACTAL', 'HYBRID_MULTIFRACTAL', 'FBM', 'HETERO_TERRAIN'] 
    musgrave_node.musgrave_type = rng.choice(type_options)





def add_random_perlin_noise(mat, seed=None):
    # Initialize the RNG with the given seed
    rng = np.random.default_rng(seed)

    tree = mat.node_tree

    # Add Noise Texture node (this will generate Perlin noise by default in Blender)
    noise_node = tree.nodes.new("ShaderNodeTexNoise")

    # Position the Noise node to the left of the Principled BSDF node for neatness
    mat_node = tree.nodes["Principled BSDF"]
    noise_node.location = (mat_node.location.x - 300, mat_node.location.y)

    # Connect the Noise Color output to the Base Color of the Principled BSDF
    links = tree.links
    link = links.new(noise_node.outputs["Color"], mat_node.inputs["Base Color"])

    # Randomly set the Noise settings using RNG
    noise_node.inputs["Scale"].default_value = rng.uniform(1.0, 10.0)
    noise_node.inputs["Detail"].default_value = rng.uniform(0.0, 10.0) # detail level of the noise
    noise_node.inputs["Distortion"].default_value = rng.uniform(0.0, 2.0) # distortion of the noise

    # Adjusting the roughness based on the noise can give interesting results
    mat_node.inputs["Roughness"].default_value = rng.uniform(0.1, 0.8)


def add_image_texture(mat, image_path, seed=None):
    # Initialize the RNG with the given seed
    rng = np.random.default_rng(seed)

    tree = mat.node_tree

    # Add an Image Texture node
    image_node = tree.nodes.new("ShaderNodeTexImage")

    # Load the image into the Image Texture node
    image_node.image = bpy.data.images.load(image_path)

    # Position the Image node to the left of the Principled BSDF node for neatness
    mat_node = tree.nodes["Principled BSDF"]
    image_node.location = (mat_node.location.x - 300, mat_node.location.y)

    # Connect the Image Color output to the Base Color of the Principled BSDF
    links = tree.links
    link = links.new(image_node.outputs["Color"], mat_node.inputs["Base Color"])

    # Optionally, you can use the RNG to vary some properties for randomness:
    # Example: adjusting the roughness randomly
    mat_node.inputs["Roughness"].default_value = rng.uniform(0.1, 0.8)
