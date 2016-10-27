#!/bin/bash
make_paras.py -g $1 -m $2

make_data.py -g $1 -m $2

cd $1

create_mge_tables.py $2 -p MGE$2 > MGE$2/stdmge

cd ..

create_ics.py $2 -f $1 -l
