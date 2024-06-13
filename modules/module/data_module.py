"""
创建不同训练过程中的 数据加载器
"""
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from data.NCaltech101.dataset import build_ncaltech
from data.NCaltech101.utils.collate import custom_collate


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, dataset_config: Dict):
        super().__init__()
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None
        self.num_workers_train = dataset_config.get("num_workers_train", None)
        self.num_workers_val = dataset_config.get("num_workers_val", None)
        self.batch_size_train = dataset_config.get("batch_size_train", None)
        self.batch_size_val = dataset_config.get("batch_size_val", None)
        assert self.num_workers_train is not None
        assert self.num_workers_val is not None
        assert self.batch_size_train is not None
        assert self.batch_size_val is not None
        self.dataset_name = dataset_name
        assert self.dataset_name is not None
        self.dataset_config = dataset_config
        self.bins = dataset_config.get("bins", None)
        assert self.bins is not None
        self.data_path = dataset_config.get("data_path", None)
        assert self.data_path is not None
        self.split_ratio = dataset_config.get("split_ratio", None)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.dataset_name == "NCaltech101":
                self.dataset_train = build_ncaltech(self.bins, self.data_path, "train", split_ratio=self.split_ratio,
                                                    transform=True)
                self.dataset_val = build_ncaltech(self.bins, self.data_path, "validation", split_ratio=self.split_ratio,
                                                  transform=True)
            else:
                raise NotImplementedError

        elif stage == "validation":
            if self.dataset_name == "NCaltech101":
                self.dataset_val = build_ncaltech(self.bins, self.data_path, "validation", split_ratio=self.split_ratio,
                                                  transform=True)
            else:
                raise NotImplementedError

        elif stage == "test":
            if self.dataset_name == "NCaltech101":
                self.dataset_test = build_ncaltech(self.bins, self.data_path, "test", split_ratio=self.split_ratio,
                                                   transform=True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # 不写特别复杂的 dataloader，写简单的加载 NCaltech101 数据集即可
        # 需要 data/NCaltech101/dataset.py 中生成的 dataset 对象 给到这里的 DataLoader
        return DataLoader(dataset=self.dataset_train,
                          batch_size=self.batch_size_train,
                          num_workers=self.num_workers_train,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=custom_collate)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.dataset_val,
                          batch_size=self.batch_size_val,
                          num_workers=self.num_workers_val,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=custom_collate)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.batch_size_val,
                          num_workers=self.num_workers_val,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=custom_collate)
