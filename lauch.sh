#! /bin/sh

EXECDIR=$PWD
RESDIR=$EXECDIR/run$(date +%d-%H%M)

EXEC="test"
N=60
D_O=0.06
RE_S="Rfrac_N=60_r0=16"
IM_S="Ifrac_N=60_r0=16"
ITV="interval.txt"
EIG=80
XTOL=100
NT=-1


mkdir $RESDIR
cp $RE_S $RESDIR/
cp $IM_S $RESDIR/
cp $ITV $RESDIR/
cd $RESDIR
../$EXEC $N $D_O $RE_S $IM_S $ITV $EIG $XTOL $NT