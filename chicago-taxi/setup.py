import setuptools

REQUIRED_PACKAGES = [
    "tensorflow-data-validation>=1.2.0, <=1.8.0"
]

setuptools.setup(
    name="executor",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"src": ["raw_schema/schema.pbtxt"]},
)
