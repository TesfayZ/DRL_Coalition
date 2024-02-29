#!/bin/sh
#SBATCH --array=00-39
#SBATCH --nodes=1
#SBATCH --job-name=Chapter2
#SBATCH --time 24:00:00
source activate vOnlineRlall
#/ECShome/tzg1e19/.conda/envs/vOnlineRlall/bin/activate
#pip install typing_extensions
#pip install mip==1.9.0
#pip install gurobipy
python main.py $SLURM_ARRAY_TASK_ID #00 pass idedx for saving file, do this 