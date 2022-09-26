import re
from typing import List


def normalize_names(names: List[str]):
    """
    Normalize list of names.

    :param names: List of input names
    """

    def _normalize_name(name: str):
        name = re.sub("[ -]", "_", name)
        return re.sub("[^A-Za-z0-9_]+", "", name).lower()

    names = [_normalize_name(name) for name in names]

    assert len(set(names)) == len(names), "Created duplicate names."

    return names
