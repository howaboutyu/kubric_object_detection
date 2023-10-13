# Makefile for building Docker image, running synthetic data generator, downloading models, and running training

# Variables
IMAGE_NAME := gpu_docker_tf2
DOCKERFILE := Dockerfile.gpu
KUBRIC_IMAGE := kubricdockerhub/kubruntu
TRAIN_SCRIPT := /tf/models/research/object_detection/model_main_tf2.py

PROMPT := "Design a texture image for a label for a beverage [beers, soft drink, coffee, etc] . Visualize a texture image representing the drink's essence. Intersperse with relevant illustrations and motifs. Overlay with bold characters symbolizing the beverage's name in [english or japanese]. A detailed nutritional table and recycling symbols and emblems at a suitable location. The theme, color palette, and design elements should harmonize with the drink's character."

# Object detection variables
PIPELINE_CONFIG_PATH := /workspace/tx-trainer/object_detector/effdet0.config
MODEL_DIR := /workspace/tx-trainer/models/effdet0
EXPORT_SCRIPT := /tf/models/research/object_detection/exporter_main_v2.py
EXPORT_DIR := $(MODEL_DIR)/exported

# Targets
build-docker:
	@sudo docker pull $(KUBRIC_IMAGE)
	@sudo docker build . --tag=$(IMAGE_NAME) -f $(DOCKERFILE)

generate-textures:
	@docker run -it \
		-v "$(shell pwd)":/kubric \
		--gpus all \
		-e CUDA_VISIBLE_DEVICES="0" \
		$(IMAGE_NAME) python /kubric/syndata_generator/sd2_texture_generator.py \
		--prompt=$(PROMPT) \
		--output_folder=generated_textures \
		--num_images=200

run-synthetic:
	@docker run -it \
		-v $(shell pwd):/kubric \
		$(KUBRIC_IMAGE) python /kubric/syndata_generator/kubric_generator.py \
		--texture-dir generated_textures  \
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


evaluate-model:
	@docker run -it \
		-v "$(shell pwd)":/workspace/tx-trainer \
		--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-e CUDA_VISIBLE_DEVICES="-1" \
		-e TF_CPP_MIN_LOG_LEVEL='0' \
		$(IMAGE_NAME) \
		python $(TRAIN_SCRIPT) \
		--pipeline_config_path=$(PIPELINE_CONFIG_PATH) \
		--model_dir=$(MODEL_DIR) \
		--checkpoint_dir=$(MODEL_DIR) \
		--sample_1_of_n_eval_examples=1 \
		--alsologtostderr

export-model:
	@docker run -it \
		-v "$(shell pwd)":/workspace/tx-trainer \
		--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-e CUDA_VISIBLE_DEVICES="-1" \
		-e TF_CPP_MIN_LOG_LEVEL='0' \
		$(IMAGE_NAME) \
		python $(EXPORT_SCRIPT) \
		--pipeline_config_path=$(PIPELINE_CONFIG_PATH) \
		--trained_checkpoint_dir=$(MODEL_DIR) \
		--output_directory=$(EXPORT_DIR)
