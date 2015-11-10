#!/usr/bin/env python

from setuptools import setup

with open("requirements.txt") as f:
    requires = "".join(f.readlines())

setup(name='Geodat',
      version='1.0',
      description='Geophysical Data Analysis Tool (GeoDAT)',
      author='Kit Yan Choi',
      author_email='kit@kychoi.org',
      url='http://kychoi.org/geodat_doc/',
      dependency_links = [
          "http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/"
          ],
      packages=["geodat","geodat.plot"],
      install_requires = requires,
      test_suite = "tests.full_suite",
     )
