IMAGE_NAME := yosshi999/virtual-background
VERSION := 0.0.1
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PREPROCESS_UPSTREAM :=

.PHONY: d_build
d_build:
	docker build \
		-t $(IMAGE_NAME):$(VERSION) \
		-f Dockerfile .

.PHONY: d_test
d_test:
	docker run --rm --gpus all \
		$(IMAGE_NAME):$(VERSION) \
		python -c "import torch;print('OK' if torch.cuda.is_available() else 'NG')"

.PHONY: run
run:
	python main.py

.PHONY: vis_data
vis_data:
	docker run --rm \
		-v $(MAKEFILE_DIR)/mlruns:/tmp/mlruns \
		-v $(MAKEFILE_DIR)/data:/opt/data \
		-v $(MAKEFILE_DIR)/fiftyone:/root/fiftyone \
		-v $(MAKEFILE_DIR)/preprocess/src:/mlflow/projects/code/src:ro \
		-p 5151:5151 \
		$(IMAGE_NAME):$(VERSION) \
		python -m src.ui --upstream /tmp/mlruns/0/$(PREPROCESS_UPSTREAM)/artifacts/downstream

.PHONY: vis_data_local
vis_data_local:
	FIFTYONE_DATASET_ZOO_DIR=./fiftyone python -m preprocess.src.ui
