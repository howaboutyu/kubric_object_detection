
# Telexistence Assignment: Detecting Cans and Bottles via Synthetic Data



## Introduction

This repository is designed to identify bottles and cans in real images by exclusively leveraging synthetic data during the training process. This endeavor unfolds into two pivotal components: synthetic data generation and object detection. Specifically, we utilize [Kubric](https://github.com/google-research/kubric) for rendering synthetic data using supplied asset geometries and employ [Stable Diffusion 2](https://huggingface.co/stabilityai) to generate random textures. Furthermore, the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is used to adeptly train our object detection model.


## Setup

### Pre-requisites: 

Ensure to have [NVIDIA Docker Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) installed on your system.

```bash
# Install 'make' utility
sudo apt install build-essential
```


### Building Docker Images: 

Execute the following command to build docker images:

```bash
make build-docker
```
The above command initiates the pull of `kubricdockerhub/kubruntu` for Kubric, subsequently building a Docker image tagged as `gpu_docker_tf2`. This image incorporates TensorFlow 2 for object detection and PyTorch.


# Discussion



## Challenges

Several challenges are encountered in this project, including:

* Some objects in the testing images are not present in the assets folder.
* Variation in cameras, lighting conditions, backgrounds, and specular highlights.
* Motion blur in the images.
* Presence of objects other than bottles or cans.
* Different environmental settings, such as items inside a fridge, on a table, or on a shelf.
* Diverse material textures, including glass, plastic, and metal.
* Varied object shapes, such as round, square, and cylindrical.
