# Synthetic Data Generation for Object Detection

The objective of this project is to generate synthetic data to train an object detector capable of identifying bottles or cans within real-world images. Several synthetic data generation tools were considered, including Nvidia Omniverse, PyBullet, and Unity. After careful evaluation, Kubric emerged as a promising choice.

You can find more information about Kubric [here](https://github.com/google-research/kubric/blob/main/challenges/multiview_matting/worker.py).

## Kubric Installation

TODO

## Kubric Output

For Kubric, the following class mapping is applied:

1. Bottles - Class ID: 1
2. Cans - Class ID: 2

## Tasks

### Kubric

- [x] Create an initial script that generates scenes with random objects placed on a plane.
- [ ] Introduce random lighting conditions and adjust camera field of view (FOV) to diversify the dataset.
- [ ] Implement random motion blur to simulate real-world scenarios.
- [ ] Incorporate random lighting variations.
- [ ] Add random distractor objects to enhance the dataset's complexity.
- [ ] Explore the use of HDRI domes for generating random background environments.
- [ ] Include textures and objects resembling cans and bottles to simulate real-world appearances.

### Kubric to TFRecords

- [ ] Develop a process to convert Kubric-generated data into TensorFlow Records (TFRecords) format, suitable for training an object detection model.
