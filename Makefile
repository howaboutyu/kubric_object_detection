# Makefile for building Docker image, running synthetic data generator, downloading models, and running training

# Variables
IMAGE_NAME := gpu_docker_tf2
DOCKERFILE := Dockerfile.gpu
KUBRIC_IMAGE := kubricdockerhub/kubruntu
TRAIN_SCRIPT := /tf/models/research/object_detection/model_main_tf2.py
PIPELINE_CONFIG_PATH := /workspace/tx-trainer/object_detector/effdet0.config
MODEL_DIR := /workspace/tx-trainer/models/effdet0

# Targets
build-docker:
	@sudo docker pull $(KUBRIC_IMAGE)
	@sudo docker build . --tag=$(IMAGE_NAME) -f $(DOCKERFILE)

run-synthetic:
	@docker run -it \
		-v $(shell pwd):/kubric \
		$(KUBRIC_IMAGE) python /kubric/syndata_generator/kubric_generator.py \
		--texture-dir syndata_generator/generated_textures  \
		--output-path kubric_synthetic_data_output \
		--num-generation 2000

download_object_detection_models:
	@mkdir -p zoo
	@wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
	@tar -xvzf efficientdet_d0_coco17_tpu-32.tar.gz -C zoo 

train:
	@docker run -it \
		-v "$(shell pwd)":/workspace/tx-trainer \
		--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-e CUDA_VISIBLE_DEVICES="0" \
		-e TF_CPP_MIN_LOG_LEVEL='0' \
		$(IMAGE_NAME) \
		python $(TRAIN_SCRIPT) \
		--pipeline_config_path=$(PIPELINE_CONFIG_PATH) \
		--model_dir=$(MODEL_DIR) \
		--sample_1_of_n_eval_examples=1 \
		--alsologtostderr
