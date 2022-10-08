from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision


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
        x = batch["img"]
        y_pred = self.model(x)

        metrics = self._calculate_metrics(y_pred, batch["target"])
        metrics["val_acc"] = torch.mean(
            (torch.argmax(y_pred, dim=1) == batch["target"]).float()
        )

        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        return metrics["val_acc"]

    def configure_optimizers(self):
        """
        Configure optimizers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
