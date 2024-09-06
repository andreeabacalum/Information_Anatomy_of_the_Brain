#!/bin/bash
#PBS -l select=1:ncpus=5:mem=40gb
#PBS -l walltime=1:00:00

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source /rds/general/user/ab5621/home/venv/example-env/bin/activate

cd $HOME/Masters-Dissertation

python3 dynamic_information.py 
