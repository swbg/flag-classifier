import logging
import os
from collections import defaultdict

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup

from flag_classifier.data_loader import load_svgs
from flag_classifier.utils import normalize_names

_logger = logging.getLogger(__name__)


def _load_index_sovereign_states(url: str) -> pd.DataFrame:
    """
    Create index of flags from Wikipedia's page on flags of sovereign states.

    Original URL: https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags

    :param url: URL to page about flag of sovereign states
    :return: Dataframe with flag name and URLs
    """
    soup = BeautifulSoup(requests.get(url).text, features="lxml")

    gallery_boxes = soup.select(
        "div#mw-content-text > div.mw-parser-output li.gallerybox"
    )
    parsed_data = defaultdict(list)

    for gallery_box in gallery_boxes:
        parsed_data["name"].append(
            gallery_box.select_one("div.gallerytext").text.strip()
        )
        parsed_data["url_wikipedia"].append(
            gallery_box.select_one("div.gallerytext a")["href"]
        )
        parsed_data["url_wikimedia"].append(gallery_box.select_one("a.image")["href"])
        parsed_data["url_file"].append(
            "https:"
            + "/".join(gallery_box.select_one("img")["src"].split("/")[:-1]).replace(
                "/thumb", ""
            )
        )

    return pd.DataFrame(parsed_data)


@click.command()
@click.option(
    "--url",
    help="URL to wikipedia page of sovereign state flags",
    required=True,
    type=click.Path(),
)
@click.option(
    "--out_dir",
    help="Output directory for data index and data",
    required=True,
    type=click.Path(),
)
def main(url, out_dir) -> None:
    """
    Download sovereign state flags.
    """
    os.makedirs(out_dir, exist_ok=True)

    index = _load_index_sovereign_states(url)

    index["file_name"] = normalize_names(index.name)
    index.to_csv(os.path.join(out_dir, "data_index.csv"), index=False)

    load_svgs(
        urls=index.url_file,
        out_names=index.file_name + ".svg",
        out_dir=os.path.join(out_dir, "svg"),
    )


if __name__ == "__main__":
    main()
