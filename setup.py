import os
from setuptools import setup
from setuptools import find_packages

setup(
    name = "gym_multi_env",
    version = "0.0.1",
    author = "Martin Seiler",
    description = ("Join multiple environments in Open AI Gym"),
    license = "",
    keywords = "openai gym multi environment",
    packages=find_packages(),
    install_requires=[
        'gym'
    ],
)