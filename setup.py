from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="seg_cl_pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-pipeline=seg_cl_pipeline.pipeline:main",
        ],
    },
    author="Moritz Schillinger",
    author_email="moritz.schillinger@fau.de",
    description="An image segmentation and analysis pipeline for MSOT research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://gitos.rrze.fau.de/ec65ohyq/pad-seg-cl-pipeline",
    python_requires=">=3.10",
)
