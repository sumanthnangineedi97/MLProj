from setuptools import setup, find_packages
from typing import List

EXCLUDE_EDITABLE = "-e ."

def load_requirements(file_path: str) -> List[str]:
    """
    Loads and cleans the list of dependencies from the given requirements file.
    """
    with open(file_path, "r") as f:
        dependencies = [line.strip() for line in f.readlines()]

    if EXCLUDE_EDITABLE in dependencies:
        dependencies.remove(EXCLUDE_EDITABLE)

    return dependencies

setup(
    name="mlproj",
    version="0.0.1",
    author="Akhila",
    author_email="nsaisumanth97@gmail.com",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt")
)
