from setuptools import setup

package_name = "" # to be changed


def read_requirements():
    with open(f"requirements.txt", "r") as f:
        return f.read().splitlines()

def read_version():
    with open("version.txt", "r") as f:
        return f.read().strip()

def read_readme():
    with open("README.md", "r") as f:
        return f.read()

setup(
    name=package_name,
    version=read_version(),
    install_requires=read_requirements(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="", # to be added
    author_email="", # to be added
    url="", # to be added
)
