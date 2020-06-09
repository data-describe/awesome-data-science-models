from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas-gbq==0.11.0',
                     'scikit-learn==0.20.2',
                     'joblib==0.14.0',
                     'six==1.15.0',
                     'hyperopt==0.2.2']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='RF trainer application',
    scripts=['predictor.py']
)
