import logging
import os
from collections import defaultdict

import click
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from flag_classifier.dataset import FlagGeneratorDataset

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--config_pth",
    help="Path to config file",
    required=True,
    type=click.Path(),
)
@click.option(
    "--override/--no-override",
    default=False,
    help="Whether to override existing epochs",
)
@click.option(
    "--num_workers",
    default=8,
    help="Number of worker threads",
)
def main(config_pth, override, num_workers) -> None:
    """
    Prepare flag training dataset.
    """
    with open(config_pth, "r") as f:
        config = yaml.safe_load(f)

    config["num_workers"] = num_workers

    ds = FlagGeneratorDataset(config, mode="train")
    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config["num_workers"],
    )

    for epoch in range(config["max_epochs"]):
        _logger.info("Preparing epoch %d...", epoch)

        data_index = defaultdict(list)
        dataset_dir = os.path.join(config["dataset_dir"], f"epoch_{epoch:05}")

        if not override and os.path.exists(os.path.join(dataset_dir, "data_index.csv")):
            _logger.info("Skipping existing epoch %d", epoch)
            continue

        os.makedirs(os.path.join(dataset_dir, "img"), exist_ok=True)

        ds.set_seed(epoch)

        for i, batch in enumerate(dl):
            file_name = f"img/img_{i:05}.pt"

            img = batch["img"][0]
            target = int(batch["target"][0])

            torch.save(img, os.path.join(dataset_dir, file_name))
            data_index["file_name"].append(file_name)
            data_index["target"].append(target)

        pd.DataFrame(data_index).to_csv(
            os.path.join(dataset_dir, "data_index.csv"), index=False
        )


if __name__ == "__main__":
    main()
