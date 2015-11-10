#!/bin/bash
cd $HOME
wget http://kychoi.org/geodat/pyferret-1.1.0-source-customed.tar.gz
gunzip pyferret-1.1.0-source-customed.tar.gz
tar -xvf pyferret-1.1.0-source-customed.tar
cd pyferret-1.1.0-source
make
make install
