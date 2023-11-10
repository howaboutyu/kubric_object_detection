
# Detecting Cans and Bottles via Synthetic Data

![cool_image](https://github.com/howaboutyu/kubric_object_detection/assets/63342319/ac33a42c-53f5-4615-bceb-cbec99fc6991)


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
# NOTE: might need sudo before the make command
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

Or, download the generated textures

```bash
wget https://storage.googleapis.com/tx-hyu-task/generated_textures.zip
unzip generated_textures.zip
```

#### Generating Scenes with Kubric

You can render random scenes with organized or random placements of cans and bottles into the scene using random HDRI domes, texture placements from SD2 textures, random textures, random camera FOV, Blender rendering settings, lighting, and camera positions. Execute the following command:

```bash
make run-synthetic
```

This will produce around 20,000 images along with bounding box information.

Or, download the generated synthetic data, which has around 26k images:

```bash
# Download generated data generated from kubric
wget https://storage.googleapis.com/tx-hyu-task/kubric_synthetic_data_output.tar
tar -xvf tfrecord_output.tar
```

#### Creating TFRecords

After generating the data, create TFRecords from it with the following command:

```bash
make create-tfrecords
```

Or, download generated tfrecords:

```bash
# Download tfrecords
wget https://storage.googleapis.com/tx-hyu-task/tfrecord_output.tar
tar -xvf tfrecord_output.tar
```

For reproducing the object detection results only `tfrecord_output.tar` is required.

## Object Detection

The object detector is based on an EfficientDet 0 model, and it does not utilize any external pretrained weights. Different datasets are used see [Discussion](#Discussion).

- **Initial Model (v1):** An initial model with an input size of 384x384 pixels was trained from scratch as a starting point for later models. 

- **Improved Model (v2):** A second model with an input size of 512x512 pixels was trained - weights are initialized from the initial model.

### Training

To reproduce the training process for either model, follow these steps:

1. Depending on the model you want to train, modify the Makefile accordingly:

```makefile
# for model v1 - 384x384
PIPELINE_CONFIG_PATH := /workspace/tx-trainer/object_detector/effdet0.config
MODEL_DIR := /workspace/tx-trainer/models/effdet0

# After training model v2 train model v2

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

## Pretrained Weights

Download pretrained weights for EfficientDet V0 for input sizes 384x384 and 512x512:

   ```bash
   wget https://storage.googleapis.com/tx-hyu-task/models.tar
   tar -xvf models.tar
   ```
To view training logs, run TensorBoard:

   ```bash
   tensorboard --logdir=models
   ```

Find the final exported TensorFlow SavedModel at:

   ```
   models/effdet0-v2/exported/saved_model
   ```


## Inference

```bash
# Download original assignment zip to get target images
wget https://storage.googleapis.com/tx-hyu-task/drink_detection_assigment.zip 
unzip drink_detection_assigment.zip

# Run inference on final model
sudo docker run -v`pwd`:/od gpu_docker_tf2 python /od/object_detector/inference.py \
                                --model_dir /od/models/effdet0-v2/exported/saved_model \
                                --image_dir /od/drink_detection_assigment/target_images/ \
                                --output_dir /od/detection_output \
                                --output_json /od/detection_output.json \
                                --score_threshold 0.5                    
```

* `--model_dir`: The directory path to the pre-trained saved model, this is created by using the `make export-model` command after training.
* `--image_dir`: The directory containing the input images on which object detection will be performed.
* `--output_dir`: The directory where the detection results will be saved.
* `--output_json`: The path to the JSON file where detection results in JSON format will be stored.
* `--score_threshold`: The probability threshold used during the filtering of the detection results.



### Json schema

The JSON output consists of a list of dictionaries, with each dictionary representing the detection results for a specific image. The schema is as follows:

* Image Path (image_path): A string representing the file path of the image for which detections were made.
* Detections (detections): A list containing dictionaries, each representing a detected object within the image. Each detection dictionary contains the following attributes:
  * Class (class): A string representing the class or category of the detected object (e.g., "bottle," "can").
  * Score (score): A floating-point number indicating the confidence score of the detection. Scores range from 0 to 1, with higher scores indicating greater confidence in the detection.
  * Bounding Box (bbox): A list of integers representing the bounding box coordinates around the detected object. The format is [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top-left corner, and (x_max, y_max) is the bottom-right corner of the bounding box.



## Results

Here are some detection outputs from the model on real-world images.

|  |  |
|---|---|
| ![005](https://github.com/howaboutyu/kubric_object_detection/assets/63342319/dbdc1b83-5883-44bf-bb23-87e397446185) | ![004](https://github.com/howaboutyu/kubric_object_detection/assets/63342319/6009efba-4fa9-44b1-8580-cff11e4db2aa) |
| ![003](https://github.com/howaboutyu/kubric_object_detection/assets/63342319/7b74534a-a30a-4ff3-b76c-50e706e2eec4) | ![002](https://github.com/howaboutyu/kubric_object_detection/assets/63342319/0fd5c9d4-eca5-4235-b0c2-8bad50b00c80) |



