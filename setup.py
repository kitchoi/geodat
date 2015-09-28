#!/usr/bin/env python

from setuptools import setup

setup(name='Geodat',
      version='1.0',
      description='Geophysical Data Analysis Tool (GeoDAT)',
      author='Kit Yan Choi',
      author_email='kit@kychoi.org',
      url='http://kychoi.org/geodat_doc/',
      packages=["geodat","geodat.plot"],
      install_requires = ["numpy","scipy","matplotlib","basemap"],
      test_suite = "tests.get_tests",
     )
