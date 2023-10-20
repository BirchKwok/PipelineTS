from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


base_reqs = read_requirements(Path('.').parent.joinpath("requirements/core.txt"))
extra_reqs = read_requirements(Path('.').parent.joinpath("requirements/extra.txt"))

all_reqs = base_reqs + extra_reqs

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PipelineTS',
    version="0.3.5",
    description='One-stop time series analysis tool, supporting time series data preprocessing, '
                'feature engineering, model training, model evaluation, and model prediction.',
    keywords='time series forecasting',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    python_requires=">=3.9",
    url='https://github.com/BirchKwok/PipelineTS',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=base_reqs,
    extras_require={"all": all_reqs, "core": base_reqs},
    zip_safe=False,
)
