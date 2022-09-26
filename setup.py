from setuptools import find_packages, setup

setup(
    name="flag_classifier",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/swbg/flag_classifier",
    author="Stefan Weissenberger",
    description="Flag image classifier.",
    zip_safe=False,
)
