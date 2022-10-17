from setuptools import setup

setup(
    name="nav_flavor",
    version="0.1",
    description="nvidia navigator flavor",
    author="Lilun Cheng",
    license="MIT",
    packages=["nav_flavor"],
    install_requires=["mlflow", "pytest", "pytest-mock"],
    # Require 3.9 before Apple M1 has trouble installing numpy for versions <= 3.8
    python_requires=">=3.8",
    zip_safe=False,
)
