import pathlib
import pkg_resources
from setuptools import setup 


with pathlib.Path('requirement.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name='colm',
    packages=["colm"],
    version='0.1',
    description='CoLM',
    author='Dang Nguyen',
    url='https://github.com/hsgser/CoLM',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)
