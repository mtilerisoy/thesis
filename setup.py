from setuptools import setup, find_packages

setup(
    name="vilt",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="0.1.0",
    license="MIT",
    description="Compressing Vision-and-Language Transformers",
    author="Mustafa Talha Ilerisoy",
    author_email="mtilerisoy@gmail.com",
    url="https://github.com/mtilerisoy/thesis",
    keywords=["vision and language compression", "transformers", "pytorch"],
    install_requires=["torch", "pytorch_lightning"],
)
