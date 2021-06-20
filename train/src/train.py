import argparse
from distutils.dir_util import copy_tree
from pathlib import Path

from omegaconf import DictConfig
import mlflow
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import cv2

def main(cfg: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=str, default="./data/preprocess")
    parser.add_argument("--downstream", type=str, default="./data/model")
    args = parser.parse_args()
    UPSTREAM = Path(args.upstream)
    DOWNSTREAM = Path(args.downstream)

    copy_tree(
        str(UPSTREAM / "fiftyone_db"),
        "/root/.fiftyone/var/lib/mongo"
    )
    DOWNSTREAM.mkdir(exist_ok=True)
    (DOWNSTREAM / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    import fiftyone
    dataset = fiftyone.load_dataset(
        cfg.dataset.name + "-val",
    )
    for x in dataset.take(2):
        fn = Path(x.filepath)
        im = Image.open(fn)
        print(im.size)
        mlflow.log_image(np.asarray(im), fn.name)
        maskAll = np.zeros((im.height, im.width), np.uint8)
        for det in x.segmentations.detections:
            relX, relY, relW, relH = det.bounding_box
            x1 = int(im.width * relX)
            x2 = int(im.width * (relX + relW))
            y1 = int(im.height * relY)
            y2 = int(im.height * (relY + relH))
            mask = np.where(det.mask, 255, 0).astype(np.uint8)
            maskAll[y1:y2, x1:x2] |= cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        mlflow.log_image(np.where(maskAll[:,:,None] == 255, np.asarray(im), 0), fn.stem + "-mask.png")

    mlflow.log_artifacts(
        str(DOWNSTREAM),
        artifact_path="downstream"
    )

if __name__ == '__main__':
    main(OmegaConf.load("config/config.yaml"))