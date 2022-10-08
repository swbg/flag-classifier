from typing import Any, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import FlagDataset


class FlagDataModule(pl.LightningDataModule):
    """
    Data module for training the flag classifier.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FlagDataModule.

        :param config: Global configuration
        """
        super().__init__()

        self.config = config
        self.dataset = FlagDataset(config)

    def train_dataloader(self) -> DataLoader:
        """
        Create train data loader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation data loader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["num_workers"],
        )
