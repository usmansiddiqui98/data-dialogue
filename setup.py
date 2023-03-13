from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


# get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()


# base requirements
install_requires = open(path.join(here, "requirements.txt")).read().strip().split("\n")

setup(
    name='data-dialogue',
    package_dir={"": "src"},
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    version='0.1.0',
    description='Whether for a B2B or B2C company, relevance and longevity in the industry depends on how well the products answer the needs of the customers. However, when the time comes for the companies to demonstrate that understanding — during a sales conversation, customer service interaction, or through the product itself — how can companies evaluate how they measure up?',
    author='Usman Siddiqui',
    license='MIT',
)
