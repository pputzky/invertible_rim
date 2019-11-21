from setuptools import setup

setup(
    name='irim',
    version='0.1.0',
    packages=['irim', 'irim.rim', 'irim.core', 'irim.test', 'irim.utils', 'examples'],
    url='https://github.com/pputzky/invertible_rim',
    license='MIT',
    author='Patrick Putzky',
    author_email='patrick.putzky@gmail.com',
    description='A library for training Recurrent Inference Machines using Invert to Learn'
)
