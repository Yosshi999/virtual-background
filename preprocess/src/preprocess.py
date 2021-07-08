import argparse
from pathlib import Path

import fiftyone
import fiftyone.core.odm
import fiftyone.zoo as foz
from omegaconf import DictConfig
import mlflow
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/preprocess/coco2014.yaml")
    parser.add_argument("--downstream", type=str, default="./data/preprocess")
    args = parser.parse_args()
    DOWNSTREAM = Path(args.downstream)
    cfg = OmegaConf.load(args.config)

    DOWNSTREAM.mkdir(exist_ok=True)
    (DOWNSTREAM / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    database_dir = fiftyone.config.database_dir
    dataset_train = foz.load_zoo_dataset(
        cfg.dataset.zoo_name,
        split="train",
        label_types=cfg.dataset.label_types,
        classes=cfg.dataset.classes,
        only_matching=True,
        seed=0,
        shuffle=True,
        dataset_name=cfg.dataset.name + "-train"
    )
    dataset_val = foz.load_zoo_dataset(
        cfg.dataset.zoo_name,
        split="validation",
        label_types=cfg.dataset.label_types,
        classes=cfg.dataset.classes,
        only_matching=True,
        shuffle=False,
        dataset_name=cfg.dataset.name + "-val"
    )
    dataset_train.persistent = True
    dataset_val.persistent = True
    fiftyone.core.odm.sync_database()

    mlflow.log_artifacts(
        str(DOWNSTREAM),
        artifact_path="downstream"
    )
    mlflow.log_artifacts(
        database_dir,
        artifact_path="downstream/fiftyone_db"
    )

if __name__ == '__main__':
    main()