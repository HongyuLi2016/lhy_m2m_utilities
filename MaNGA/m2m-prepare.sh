#!/bin/bash
m2m-make_paras.py -g $1 -m $2

m2m-make_manga_data.py -g $1 -m $2

cd $1

create_mge_tables.py $2 -p MGE$2 > MGE$2/stdmge

cd ..

#m2m-create_ics.py $2 -f $1 -l # use flat z
m2m-create_ics.py $2 -f $1 # use logz
