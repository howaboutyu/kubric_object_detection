
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




## Synthetic Data Generation

### Generating Synthetic Data

#### Texture Generation

First, generate synthetic textures using Stable Diffusion 2 by specifying the `PROMPT` in the `Makefile` and running the following command:

```bash
make generate-textures
```

This will generate approximately 10,000 labeled texture images.

#### Generating Scenes with Kubric

You can render random scenes with organized or random placements of cans and bottles into the scene using random HDRI domes, texture placements from SD2 textures, random camera FOV, Blender rendering settings, lighting, and camera positions. Execute the following command:

```bash
make run-synthetic
```

This will produce around 20,000 images along with bounding box information.

#### Creating TFRecords

After generating the data, create TFRecords from it with the following command:

```bash
make create-tfrecords
```

### Downloading Data


## Object Detection

The object detector is based on an EfficientDet 0 model, and it does not utilize any external pretrained weights. Two object detectors were trained during this project, both models undergoing 300,000 training steps with a batch size of 8. However, it's important to note that the data used and input sizes for these models were different.

- **Initial Model (v1):** An initial model with an input size of 384x384 pixels was trained from scratch as a starting point for later models. This model was trained using approximately 9,000 images.

- **Improved Model (v2):** A second model with an input size of 512x512 pixels was trained. It was initialized from the weights of the 384x384 model and used a larger dataset of approximately 16,000 images.

### Reproduction

To reproduce the training process for either model, follow these steps:

1. Depending on the model you want to train, modify the Makefile accordingly:

```makefile
# for model v1 - 384x384
PIPELINE_CONFIG_PATH := /workspace/tx-trainer/object_detector/effdet0.config
MODEL_DIR := /workspace/tx-trainer/models/effdet0

# for model v2 - 512x512
PIPELINE_CONFIG_PATH := /workspace/tx-trainer/object_detector/effdet0-v2.config
MODEL_DIR := /workspace/tx-trainer/models/effdet0-v2
```

2. After making the necessary configuration changes, execute the following commands to perform training, evaluation, and model export:

```bash
make train
make evaluate-model
make export-model
```

This process will train the selected model, evaluate its performance, and export the trained model for future use.



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
