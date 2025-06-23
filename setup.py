import os
from pathlib import Path

from setuptools import find_packages, setup

__version__ = "0.1"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open(os.path.join(this_directory, "requirements.txt"), "r") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

print(f"install_requires: {install_requires}")

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/wizard-III/ArcherCodeR",
    license="Apache 2.0",
    author="",
    author_email="",
    description="",
    install_requires=install_requires,
    # extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",