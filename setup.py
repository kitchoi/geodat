#!/usr/bin/env python

from setuptools import setup

with open("requirements.txt") as f:
    requires = f.readlines()

requires = [ req.replace("\n","") for req in requires ]
requires = [ req for req in requires if req ]


setup(name='Geodat',
      version='1.0',
      description='Geophysical Data Analysis Tool (GeoDAT)',
      author='Kit Yan Choi',
      author_email='kit@kychoi.org',
      url='http://kychoi.org/geodat_doc/',
      packages=["geodat","geodat.plot"],
      install_requires = requires,
      #test_suite = "tests.get_tests",
     )
