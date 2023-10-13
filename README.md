
# Telexistence Assignment: Detecting Cans and Bottles via Synthetic Data



## Introduction

Welcome to the repository for our object detection project, specifically aimed at identifying bottles and cans utilizing synthetic data. The project is bifurcated into two critical components: synthetic data generation and object detection. Specifically we use [Kubric](

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

