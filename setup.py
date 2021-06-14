
from setuptools import setup, find_packages

__version__ = ""
exec(open('goodok_mlds/version.py').read())


setup(
    name='goodok_mlds',
    version=__version__,
    description='Common ML datasets',
    author='Alexey U. Gudchenko',
    author_email='proga@goodok.ru',
    url='https://github.com/goodok/ml_datasets',
    packages=find_packages(),
    python_requires='>=3.6',
)
