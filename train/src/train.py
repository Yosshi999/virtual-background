import argparse
from distutils.dir_util import copy_tree
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
import numpy as np
# from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import jaccard_score

from src.data import Dataset, imagenet_denorm

class MyModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vals = []
        self.ious = []
        self.gstep = 0
    
    def prepare_data(self):
        import fiftyone
    
    def setup(self, stage=None):
        import fiftyone
        dataset_train = fiftyone.load_dataset(
            self.cfg.dataset.name + "-train",
        )
        dataset_val = fiftyone.load_dataset(
            self.cfg.dataset.name + "-val",
        )
        self.model = smp.create_model(**self.cfg.model)
        self.ds_train = Dataset(dataset_train)
        self.ds_val = Dataset(dataset_val, False)
        self.loss_fn = nn.BCEWithLogitsLoss()
        mlflow.log_param("train_size", len(dataset_train))
        mlflow.log_param("val_size", len(dataset_val))
        print(f"train: {len(dataset_train)} samples")
        print(f"val: {len(dataset_val)} samples")
    
    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=16,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=16,
            pin_memory=True
        )

    def training_step(self, batch, batch_idx):
        self.gstep += 1
        x, t = batch
        y = self(x)
        loss = self.loss_fn(y, t)
        mlflow.log_metric("loss", loss.cpu().detach().item(), self.gstep)
        return loss

    def on_validation_start(self) -> None:
        self.vals = []
        self.ious = []
    def on_validation_end(self) -> None:
        mlflow.log_metric("val_loss", np.mean(self.vals), self.gstep)
        mlflow.log_metric("mean_iou", np.mean(self.ious), self.gstep)
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.loss_fn(y, t)
        self.vals.append(loss.cpu().detach().item())
        y_bin = np.where(y.cpu().detach().numpy() > 0, 1, 0)
        t = t.cpu().detach().numpy()
        for _t, _y_bin in zip(t, y_bin):
            self.ious.append(jaccard_score(_t.reshape(-1), _y_bin.reshape(-1)))
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.cfg.optim.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, self.cfg.optim.lr_max,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=len(self.train_dataloader())),
                "interval": "step",
            },
        }
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-3)

class PredictImageCallback(Callback):
    def __init__(self, model: LightningModule):
        super().__init__()
        self.model = model
    def on_train_start(self, trainer, pl_module):
        # visualize one batch from val #
        xs, ts = next(iter(self.model.val_dataloader()))
        for i, (x, t) in enumerate(zip(xs, ts)):
            arr = x.numpy().transpose(1,2,0)
            arr = (imagenet_denorm(arr) * 255).astype(np.uint8)
            mlflow.log_image(arr, f"start/val-{i:02d}.png")
            mlflow.log_image(np.where(t[0,:,:,None] == 1, arr, 0), f"start/val-{i:02d}-mask.png")
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        # visualize one batch from predicted val #
        xs, ts = next(iter(self.model.val_dataloader()))
        with torch.no_grad():
            ys = self.model(xs.to(self.model.device)).cpu().numpy()
        for i, (x, y, t) in enumerate(zip(xs, ys, ts)):
            arr = x.numpy().transpose(1,2,0)
            arr = (imagenet_denorm(arr) * 255).astype(np.uint8)
            mlflow.log_image(arr, f"end/val-{i:02d}.png")
            mlflow.log_image(np.where(y[0,:,:,None] > 0, arr, 0), f"end/val-{i:02d}-pred.png")
            mlflow.log_image(np.where(t[0,:,:,None] == 1, arr, 0), f"end/val-{i:02d}-mask.png")
        print("Training is done.")

class ValEveryNSteps(Callback):
    def __init__(self, every_n_step, model):
        self.every_n_step = every_n_step
        self.model = model
    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.validate(model=self.model)

class SaveEveryNSteps(Callback):
    def __init__(self, every_n_step, model):
        self.every_n_step = every_n_step
        self.model = model
    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            mlflow.pytorch.log_model(self.model, f"model-{trainer.global_step}")

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/train/resnet18.yaml")
    parser.add_argument("--upstream", type=str, default="./data/preprocess")
    parser.add_argument("--downstream", type=str, default="./data/model")
    args = parser.parse_args()
    UPSTREAM = Path(args.upstream)
    DOWNSTREAM = Path(args.downstream)
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.load(UPSTREAM / "config.yaml")
    )

    copy_tree(
        str(UPSTREAM / "fiftyone_db"),
        "/root/.fiftyone/var/lib/mongo"
    )
    DOWNSTREAM.mkdir(exist_ok=True)
    (DOWNSTREAM / "config.yaml").write_text(OmegaConf.to_yaml(cfg))


    model = MyModule(cfg)
    # logger = MLFlowLogger(tracking_uri="file:/tmp/mlruns")
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=cfg.train.epoch,
        logger=False,
        callbacks=[
            SaveEveryNSteps(cfg.train.save_step, model),
            PredictImageCallback(model),
        ])
    trainer.fit(model)
    mlflow.pytorch.log_model(model, f"model-last")

    # save #
    mlflow.log_artifacts(
        str(DOWNSTREAM),
        artifact_path="downstream"
    )

if __name__ == '__main__':
    main()