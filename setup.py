from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PipelineTS',
    version="0.1.5",
    description='One-stop time series analysis tool, supporting time series data preprocessing, '
                'feature engineering, model training, model evaluation, and model prediction.',
    keywords='timeseries machine learning',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    url='https://github.com/BirchKwok/PipelineTS',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=[
        'scikit-learn>=1.3.0',
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'matplotlib>=3.7.1',
        'frozendict>=2.3.0',
        'darts>=0.24.0',
        'prophet>=1.1.4',
        'spinesTS>=0.2.5',
        'spinesUtils>=0.3.5',
        'lightgbm>=3.3.5',
        'IPython>=8.12.1',
        'tabulate>=0.8.9',
        'torch>=2.1.0'
    ],
    zip_safe=False,
    include_package_data=True
)