#!/bin/bash
#PBS -N PCAM
#PBS -l nodes=2:ppn=24
#PBS -o PCAMOUT.log
#PBS -j oe

module avail
module avail annaconda

cd $PBS_O_WORKDIR
module load anaconda3 

python ./test3.py




