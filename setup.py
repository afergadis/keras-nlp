from setuptools import setup, find_packages

setup(
    name='keras_nlp',
    version='0.1',
    url='',
    license='',
    author='Aris Fergadis',
    author_email='aris.fergadis@protonmail.com',
    description='Library for NLP processing for Keras models.',
    install_requires=['scikit-learn', 'numpy', 'keras'],
    packages=find_packages(),
    test_suite='keras_nlp.tests',
)
