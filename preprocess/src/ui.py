from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz
from omegaconf import DictConfig, OmegaConf
from PIL import Image

def main(cfg: DictConfig):
    dataset = foz.load_zoo_dataset(
        cfg.dataset.zoo_name,
        split="validation",
        label_types=cfg.dataset.label_types,
        classes=cfg.dataset.classes,
        max_samples=100,
        seed=0,
        shuffle=False,
        dataset_name=cfg.dataset.name
    )
    if True:
        session = fo.launch_app(dataset)
    else:
        for x in dataset.take(2):
            det = x.segmentations.detections[0]
            mask = det.mask
            w, h = Image.open(x.filepath).size
            print(x.filepath, mask.shape)
            print(w, h)
            print(h * (det.bounding_box[3]))
            print(w * (det.bounding_box[2]))
            print(mask.shape[0] / mask.shape[1], (h * (det.bounding_box[3])) / (w * (det.bounding_box[2])))
    input("> enter anything to quit")

if __name__ == '__main__':
    main(OmegaConf.load('config/config.yaml'))
