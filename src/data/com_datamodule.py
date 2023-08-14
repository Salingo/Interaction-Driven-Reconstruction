from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from src.data.components.com_dataset import ComDataset


class ComDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        com_data_dir: str,
        motion_dir: str,
        state_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train:
            self.data_train = ComDataset(
                data_dir=self.hparams.data_dir,
                com_data_dir=self.hparams.com_data_dir,
                motion_dir=self.hparams.motion_dir,
                data_index_path=self.hparams.state_dir + "/train.txt"
            )

        if not self.data_val:
            self.data_val = ComDataset(
                data_dir=self.hparams.data_dir,
                com_data_dir=self.hparams.com_data_dir,
                motion_dir=self.hparams.motion_dir,
                data_index_path=self.hparams.state_dir + "/val.txt"
            )

        if not self.data_test:
            self.data_test = ComDataset(
                data_dir=self.hparams.data_dir,
                com_data_dir=self.hparams.com_data_dir,
                motion_dir=self.hparams.motion_dir,
                data_index_path=self.hparams.state_dir + "/test.txt"
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
