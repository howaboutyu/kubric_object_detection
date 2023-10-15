
# Telexistence Assignment: Detecting Cans and Bottles via Synthetic Data

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/howaboutyu/telexistence_assignment/assets/63342319/dc635088-ecd8-4fb9-8744-979ae975d27b" alt="cool_image" height="300px">
    <img src="https://github.com/howaboutyu/telexistence_assignment/assets/63342319/2787769a-a3de-4a79-8a4a-2dcabb90f345" alt="download" height="300px">
</div>

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


# Report 


## System Description

This project was conducted on a remote computer with the following specifications:
```bash
CPU: AMD Ryzen 7 2700X Eight-Core Processor
GPU: NVIDIA RTX 2080ti and GeForce 1080
RAM: 32GB
```


## <a name="Discussion"></a> Method

### Synthetic Data
Synthetic data was generated using Kubric with random domain randomization techniques, as outlined above. Kubric rendering was relatively slow, taking approximately 2 minutes to render 10 images per session, which might be due to the current Kubric Docker image not supporting GPU rendering. Additionally, the random object placement function currently positions objects on a plane without considering the physics of placement. As a result, objects are sometimes placed in physically impossible configurations, such as overlapping, which diminishes the realism of the synthetic data.

A total of 2600 Kubric sessions were generated, each containing 10 images, resulting in a total of 26,000 images.

### Object Detection

An initial model was trained with around 9k images, as the data generation pipeline is somewhat slow. Training consisted of 300k steps, using an input size of 384x384, a batch size of 8, and cosine learning rate decay. This model was trained from scratch without loading any pretrained weights. The configuration file for this model can be found at `object_detector/effdet0.config`. During the training of this model, the Kubric data generation reached approximately 26k images. Subsequently, another EfficientDet V0 model was trained for 50k steps with a shape of 512x512, initialized with weights from the 384x384 model. The configuration for the second model can be found at `object_detector/effdet0-v2.config`.


## Results

The results obtained from the EfficientDet V0 model with a 512x512 input resolution, using a probability threshold of 0.5 for bounding boxes, are displayed below. Notably, there are instances of misclassification for certain classes, and in some cases, objects are not accurately detected. It is worth noting that the detector's performance is somewhat suboptimal in specific images. This could potentially be attributed to differences in lighting conditions between the synthetic data used for training and the real-world target images. 

| ![017](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/e0450aa7-671a-4c2c-89a9-c4607664ebdb) | ![016](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/6b685eef-acc5-4345-afb4-72e51e785fba) |
|---|---|
| ![015](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/b4fe73a1-c516-430a-9d1a-791fc9ad1aca) | ![014](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/41fb2c90-0fdd-41a3-aaff-d0feea71058b) |
| ![013](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/0796241d-317c-4231-aa2e-2ddc5429f49a) | ![012](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/8438426d-640a-455a-b731-553f3d07e42a) |
| ![011](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/1293cfed-f157-4589-b59f-70079e275832) | ![010](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/d6b48b37-47d6-4063-a6ae-e47585ed0298) |
| ![009](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/91a1b2c3-bc89-48e6-9fb5-e77ee20b46c0) | ![008](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/b701662f-f85e-4b61-bae5-1821d2720351) |
| ![007](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/53740d36-7f50-4d70-b673-36dc11a99331) | ![006](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/9e92ddf7-2bdc-4ae4-bd03-60c2191205d2) |
| ![005](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/08d1c403-87e3-4a67-a6e8-0bc523db73fe) | ![004](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/7f25f153-bf8d-4e99-a3f4-4a7fa8b7fd0a) |
| ![003](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/8884760d-5a96-4814-838d-9c048977263b) | ![002](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/70883120-cc71-48a5-8be4-1bf68b97db4d) |
| ![001](https://github.com/howaboutyu/telexistence_assignment/assets/63342319/01d0ec86-669d-4b9f-b59f-c24bba30cf8a) | |



## Discussion
In my approach to utilizing synthetic data for training object detection models, creating realistic images closely resembling target images was essential. I employed Stable Diffusion to introduce random textures to bottle and can assets, which helped the detector focus on learning object shapes rather than being overly influenced by texture details.

However, it's worth noting a specific issue in the results section: a bottle of Coca-Cola was misclassified as a can. This misclassification might be attributed to the usage of a Coke can in the original assets, which were exclusively used for generating can bounding boxes. To mitigate such issues in the future, it might be beneficial to avoid using textures derived from the original assets.

Moreover, further improvements can be made by expanding the dataset. Increasing the number of generated images beyond the current 27k and introducing more variation in lighting conditions, shapes, and object poses could enhance the model's accuracy.

## Future Work
Future investigations could delve into generating higher-resolution images using Kubric, incorporating a wider array of shape variations, and developing more realistic environments within Kubric. Additionally, exploring training with other detection models, such as EfficientDet V1 to V6, or instance segmentation algorithms, and experimenting with various augmentation techniques and generative AI methods could be considered for potential research avenues.
