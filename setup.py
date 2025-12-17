from setuptools import setup, find_packages

package_name = "sparse_bench"


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
    packages=find_packages(exclude=["tests", "scripts"]),
    version=read_version(),
    install_requires=read_requirements(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Shenggui Li",
    author_email="somerlee.9@gmail.com",
    url="https://github.com/FrankLeeeee/Sparse-Attention-Video-Bench",
)
