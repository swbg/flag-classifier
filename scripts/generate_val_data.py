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
    "--num_workers",
    default=8,
    help="Number of worker threads",
)
def main(config_pth, num_workers) -> None:
    """
    Prepare flag validation dataset.
    """
    with open(config_pth, "r") as f:
        config = yaml.safe_load(f)

    config["num_workers"] = num_workers

    ds = FlagGeneratorDataset(config, mode="val")
    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config["num_workers"],
    )

    data_index = defaultdict(list)
    dataset_dir = os.path.join(config["dataset_dir"], "val")

    os.makedirs(os.path.join(dataset_dir, "img"), exist_ok=True)

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
