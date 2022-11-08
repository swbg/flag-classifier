import logging
import os
from collections import defaultdict
from glob import glob

import albumentations as A
import click
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2

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
    "--input_pth",
    help="Path to input data",
    required=True,
    type=click.Path(),
)
def main(config_pth, input_pth) -> None:
    """
    Prepare flag test dataset.
    """
    with open(config_pth, "r") as f:
        config = yaml.safe_load(f)

    data_index_train = pd.read_csv(config["data_index_flags"])
    data_index_train["target"] = np.arange(len(data_index_train))

    output_pth = os.path.join(config["dataset_dir"], "test")
    os.makedirs(os.path.join(output_pth, "img"), exist_ok=True)

    data_index_test = defaultdict(list)
    file_counter = 0

    aug = A.Compose(
        [
            A.Resize(config["input_size"], config["input_size"]),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0
            ),
            ToTensorV2(),
        ]
    )

    for dirr in sorted(glob(os.path.join(input_pth, "*"))):
        if not os.path.isdir(dirr):
            continue

        target = data_index_train[data_index_train.name == dirr.split("/")[-1]]

        if len(target) != 1:
            raise ValueError(f"Found {len(target)} entries for {dirr.split('/')[-1]!r}")

        target = target.iloc[0].target

        for src_file in sorted(glob(os.path.join(dirr, "*"))):
            dst_file_name = os.path.join("img", f"img_{file_counter:05}.pt")

            img = cv2.imread(src_file)[..., ::-1] / 255.0
            img = aug(image=img)["image"]
            torch.save(img, os.path.join(output_pth, dst_file_name))

            file_counter += 1

            data_index_test["file_name"].append(dst_file_name)
            data_index_test["target"].append(target)

    pd.DataFrame(data_index_test).to_csv(
        os.path.join(output_pth, "data_index.csv"), index=False
    )


if __name__ == "__main__":
    main()
