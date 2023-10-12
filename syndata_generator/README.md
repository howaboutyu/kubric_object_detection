# Synthetic Data Generation for Object Detection

The goal of this project is to generate synthetic data for training an object detector capable of identifying bottles or cans within real-world images. Kubric is a 3D simulation platform with an easy-to-use Python API that can import OBJ files with various textures. After simulating the scenes, Kubric can export useful information such as bounding boxes and segmentation masks.


[kubric github](https://github.com/google-research/kubric/blob/main/challenges/multiview_matting/worker.py).

## Kubric Installation

TODO

## Usage

TODO


## Tasks

### Kubric

- [x] Create an initial script that generates scenes with random objects placed on a plane.
- [x] Introduce random lighting conditions and adjust camera field of view (FOV) to diversify the dataset.
- [ ] Implement random motion blur to simulate real-world scenarios.
- [x] Incorporate random lighting variations.
- [ ] Add random distractor objects to enhance the dataset's complexity.
- [x] Explore the use of HDRI domes for generating random background environments.
- [x] Generate textures using generative AI methods. 

### Kubric to TFRecords

- [x] Develop a process to convert Kubric-generated data into TensorFlow Records (TFRecords) format, suitable for training an object detection model.
