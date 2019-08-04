import codecs
import os
import re
import subprocess
from setuptools import setup, find_packages


def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)


def get_contents(*args):
    """Get the contents of a file relative to the source distribution directory."""
    with codecs.open(get_absolute_path(*args), 'r', 'UTF-8') as handle:
        return handle.read()


def get_version(*args):
    """Extract the version number from a Python module."""
    contents = get_contents(*args)
    metadata = dict(re.findall('__([a-z]+)__ = [\'"]([^\'"]+)', contents))
    return metadata['version']


version = get_version('insightface', '__init__.py')
cwd = os.path.dirname(os.path.abspath(__file__))
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except subprocess.CalledProcessError:
    pass


setup(
    name='insightface',
    version=version,
    author='Evgeny Nizhibitsky',
    url='https://github.com/nizhib/pytorch-insightface/',
    description='Pretrained insightface models ported to pytorch',
    license='MIT',

    packages=find_packages(),
    install_requires=[
        'torch>=0.4.1',
        'torchvision>=0.3.0'
    ],
    python_requires='>=3.5.0'
)
