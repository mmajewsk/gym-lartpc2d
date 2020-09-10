from setuptools import setup, find_packages
from lartpc_game.version import VERSION

with open("readme.md", "r") as fh:
    long_description = fh.read()
    setup(
        name='gym_lartpc',
        version=VERSION,
        author="Maciej Majewski",
        author_email="mmajewsk@cern.ch",
        description="Openai-gym compatible reinforcement learning env for lartpc data",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://githum.com/mmajewsk/gym-lartpc2d",
        packages=find_packages(),
        install_requires=[
                "pandas>=1.0.0",
                "scipy>=1.4.1",
                "opencv_python>=4.2.0.32",
                "numpy>=1.18.1",
                "matplotlib>=3.1.3",
                "nptyping>=1.3.0",
                "gym"
        ]

    )
