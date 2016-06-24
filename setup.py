from setuptools import setup

import ghostwriter

setup(
    name="ghostwriter",
    version=ghostwriter.__version__,
    packages=['ghostwriter'],
    url="https://github.com/wpm/ghostwriter",
    license="",
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Machine-Assisted Writing",
    entry_points={
        "console_scripts": ["ghostwriter=ghostwriter.main:main"],
    },
    install_requires=[
        "tensorflow",
        "nltk"
    ]
)
