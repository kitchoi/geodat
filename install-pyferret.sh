#!/bin/bash

if [ -d "$HOME/pyferret-1.1.0-source/pyferret_install" ]; then
    if [[ $* != *--update* ]]; then
	echo "Cached pyferret_install exists.  Passed.";
	exit 0;
    fi
fi

echo "Install PyFerret"

cd $HOME
gunzip pyferret-1.1.0-source-customed.tar.gz
tar -xf pyferret-1.1.0-source-customed.tar
cd pyferret-1.1.0-source
make -s
make install -s
