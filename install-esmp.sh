if [ -d "$HOME/esmp.ESMF_6_3_0rp1_ESMP_01" ]; then
    if [[ $* != *--update* ]]; then
	echo "Cached ESMP exists.  Passed.";
	exit 0;
    fi
fi

echo "Install ESMF/ESMP"

cd $HOME
gunzip esmp.ESMF_6_3_0rp1_ESMP_01.tar.gz
tar -xf esmp.ESMF_6_3_0rp1_ESMP_01.tar
cd esmp.ESMF_6_3_0rp1_ESMP_01/esmf
export ESMF_DIR=$PWD
export ESMF_INSTALL_PREFIX=$HOME/esmf
make -s
make install -s

cd ../ESMP
export ESMFMKFILE=$HOME/esmp.ESMF_6_3_0rp1_ESMP_01/esmf/lib/libO/Linux.gfortran.64.mpiuni.default/esmf.mk
make build -s
