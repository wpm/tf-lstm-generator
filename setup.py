from setuptools import setup

import ghostwriter

setup(
    name='ghostwriter',
    version=ghostwriter.__version__,
    packages=['ghostwriter'],
    url='',
    license='',
    author='W.P. McNeill',
    author_email='billmcn@gmail.com',
    description='Machine-Assisted Writing',
    entry_points={
        'console_scripts': ['ghostwriter=ghostwriter.main:main'],
    }
)
