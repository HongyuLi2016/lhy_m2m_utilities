#!/bin/bash
make_paras.py -g $1 -m $2

make_data.py -g $1 -m $2

mkdir -p $1/MGE$2

#choose which MGE to use, total density or stellar density only, please mind to change make_paras.py as well
#ext3dmge  -o$1/MGE$2 -g$1/auxiliary_data/mge_params_$2 > $1/MGE$2/stdMGE.log
ext3dmge  -o$1/MGE$2 -g$1/auxiliary_data/pot_params_$2 > $1/MGE$2/stdMGE.log

mv $1/MGE$2/mge_pf $1/MGE$2/mge_pf_old

cp $1/auxiliary_data/mge_pf_$2 $1/MGE$2/mge_pf

create_ics.py $2 -f $1 -l
