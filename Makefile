IMAGE_NAME := yosshi999/virtual-background
VERSION := 0.0.1

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
	python -m preprocess.src.ui
