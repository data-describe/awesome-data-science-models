from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['h5py==2.10.0',
                     'six==1.15.0',
                     'scikit-learn==0.20.2',
                     'joblib==0.14.0',
                     'scikit-learn==0.20.2',
                     'opencv-python',
                     'tensorflow-datasets',
                     'gcsfs'
                    ]

setup(
    name='ml_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='version trainer application'
)