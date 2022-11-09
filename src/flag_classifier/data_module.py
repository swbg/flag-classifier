from typing import Any, Dict, Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from flag_classifier.dataset import FlagGeneratorDataset, FlagReaderDataset


class FlagDataModule(pl.LightningDataModule):
    """
    Data module for training the flag classifier.
    """

    def __init__(
        self, config: Dict[str, Any], mode: Literal["read", "generate"] = "read"
    ):
        """
        Initialize FlagDataModule.

        :param config: Global configuration
        """
        super().__init__()

        self.config = config
        self.mode = mode

        if mode == "read":
            self.train_dataset = FlagReaderDataset(config, mode="train")
            self.val_dataset = FlagReaderDataset(config, mode="val")
        else:
            self.train_dataset = FlagGeneratorDataset(config, mode="train")
            self.val_dataset = FlagGeneratorDataset(config, mode="val")

        self.test_dataset = FlagReaderDataset(config, mode="test")

    def train_dataloader(self) -> DataLoader:
        """
        Create train data loader.
        """
        return DataLoader(
            dataset=self.train_dataset,
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
            dataset=self.val_dataset,
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
            dataset=self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["num_workers"],
        )
