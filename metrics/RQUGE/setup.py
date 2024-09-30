from io import open

from setuptools import find_packages, setup

setup(
    name="rquge",
    version="0.1",
    author="Alireza Mohammadshahi*, Thomas Scialom, Majid Yazdani, Pouya Yanki, Angela Fan, James Henderson, Marzieh Saeidi",
    author_email="alireza.msh1373@gmail.com",
    description="PyTorch implementation of RQUGE score",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="BERT NLP deep learning QG QA metric",
    license="MIT",
    url="https://github.com/alirezamshi/RQUGE",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "sentencepiece",
    ],
    entry_points={
        "console_scripts": [
            "rquge=rquge_score_cli.scorer_cli:main",
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
