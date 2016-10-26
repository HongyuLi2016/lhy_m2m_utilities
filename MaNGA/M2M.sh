#!/bin/bash
mpiexec -np $2 execm2m.py  -orst_$1 $1 >std$1
cp $1.cfg rst_$1
eor_mean_chi2.py rst_$1 >> std$1
eor_entropy.py rst_$1 >> std$1
eor_obs.py rst_$1 > rst_$1/observables/repro_analysis
eor_energy.py rst_$1 > rst_$1/particles/energy_analysis
eor_weights.py rst_$1 > rst_$1/particles/weights_analysis
mv std$1 rst_$1
plotconv.py rst_$1 >> rst_$1/std$1
plotvrms.py rst_$1 -m IFU_vel -d IFU_disp

