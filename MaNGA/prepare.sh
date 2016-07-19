#!/bin/bash
make_paras.py -g $1 -m $2

make_data.py -g $1 -m $2

mkdir -p $1/MGE$2

ext3dmge  -o$1/MGE$2 -g$1/auxiliary_data/mge_params_$2 > $1/MGE$2/stdMGE.log

create_ics.py $2 -f $1
