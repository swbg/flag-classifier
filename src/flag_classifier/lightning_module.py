from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from flag_classifier.dataset import FlagReaderDataset


class FlagClassifier(pl.LightningModule):
    """
    Lightning module for training the flag classifier.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FlagClassifier.

        :param config: Global config
        """
        super().__init__()

        self.config = config

        self.model = self._build_model()
        self.loss = nn.CrossEntropyLoss()

        self.unfreeze()
        self.freeze_backbone()

    def _build_model(self):
        mobilenet = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # Reset weights of classifier
        mobilenet.classifier[0].reset_parameters()
        # Replace last linear layer with custom linear layer
        mobilenet.classifier[3] = nn.Linear(
            in_features=mobilenet.classifier[3].in_features,
            out_features=self.config["n_classes"],
            bias=True,
        )
        return mobilenet

    def unfreeze(self):
        """
        Unfreeze all parameters.
        """
        for p in self.model.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        """
        Freeze all backbone (non-classifier) parameters.
        """
        for p in self.model.features.parameters():
            p.requires_grad = False

    def _calculate_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Dict[str, Any]:
        return {"loss": self.loss(y_pred, y_true)}

    def _calculate_accuracies(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Dict[str, Any]:
        result = {}

        _, idx = y_pred.topk(5)
        for i in (1, 3, 5):
            result[f"top{i}acc"] = (
                (idx[:, :i] == y_true.view(-1, 1)).sum(axis=1).float().mean()
            )

        return result

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Perform training step.

        :param batch: Training batch
        :param batch_idx: Batch index
        """
        x = batch["img"]
        y_pred = self.model(x)

        metrics = self._calculate_metrics(y_pred, batch["target"])
        self.log_dict(metrics)

        return metrics["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Perform validation step.

        :param batch: Validation batch
        :param batch_idx: Batch index
        """
        y_pred = self.model(batch["img"])

        metrics = self._calculate_metrics(y_pred, batch["target"])
        metrics.update(self._calculate_accuracies(y_pred, batch["target"]))

        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        return metrics["top1acc"]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Perform test step.

        :param batch: Validation batch
        :param batch_idx: Batch index
        """
        y_pred = self.model(batch["img"])

        metrics = self._calculate_metrics(y_pred, batch["target"])
        metrics.update(self._calculate_accuracies(y_pred, batch["target"]))

        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        return metrics["top1acc"]

    def configure_optimizers(self):
        """
        Configure optimizers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        return optimizer

    def on_train_epoch_start(self) -> None:
        """
        Hadle train epoch start.
        """
        dataset = self.trainer.train_dataloader.dataset.datasets
        if isinstance(dataset, FlagReaderDataset):
            # Change directory of dataset
            dataset.set_up_epoch(self.current_epoch)
