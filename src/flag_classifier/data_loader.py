import logging
import os
import time
from typing import List

import requests
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def load_svgs(
    urls: List[str],
    out_names: List[str],
    out_dir: str,
    sleep: float = 0.1,
    reload: bool = False,
) -> None:
    """
    Download SVGs according to data index.

    :param urls: List of URLs pointing to SVGs
    :param out_names: Output file names
    :param out_dir: Output directory
    :param sleep: How long to sleep between requests (simple rate limiting)
    :param reload: Whether to reload if file already exists
    """
    for url, name in tqdm(zip(urls, out_names), ncols=100):
        out_pth = os.path.join(out_dir, name)
        if os.path.exists(out_pth) and not reload:
            continue

        time.sleep(sleep)

        response = requests.get(url)
        if response.status_code == 200:
            with open(out_pth, "w") as f:
                f.write(response.text)
        else:
            _logger.warning(
                "Could not download %s (Error %d), skipping.", url, response.status_code
            )
