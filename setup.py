# setup.py
from setuptools import setup, find_packages

setup(
    name="rota_aco",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "osmnx",
        "networkx",
        "matplotlib",
        "folium",
        "pytest"
    ],
)
