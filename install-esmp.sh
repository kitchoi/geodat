cd $HOME
wget http://kychoi.org/geodat/esmp.ESMF_6_3_0rp1_ESMP_01.tar.gz
gunzip esmp.ESMF_6_3_0rp1_ESMP_01.tar.gz
tar -xf esmp.ESMF_6_3_0rp1_ESMP_01.tar
cd esmp.ESMF_6_3_0rp1_ESMP_01/esmf
export ESMF_DIR=$PWD
export ESMF_INSTALL_PREFIX=$HOME/esmf
gmake
gmake install

ls $HOME/esmf/lib/lib0/*/esmf.mk

cd ../ESMP
export ESMFMKFILE=$HOME/esmf/lib/libO/Linux.gfortran.64.mpiuni.default/esmf.mk
gmake build
