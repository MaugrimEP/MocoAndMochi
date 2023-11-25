#!/bin/bash

#SBATCH -J "mochi"
#SBATCH --time 48:00:00
#SBATCH --mem 100000
#SBATCH --partition gpu_v100
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task=7
#SBATCH --output /home/2022022/tmayet02/logs/osvm/%J_%x.out
#SBATCH --error  /home/2022022/tmayet02/logs/osvm/%J_%x.err

PROJECTNAME=osvm
module purge
module load python3-DL/torch/1.12.1-cuda10.2
export PYTHONUSERBASE=~/packages/$PROJECTNAME
PATH=$PATH:~/packages/$PROJECTNAME
export PATH

cp -R . $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR

echo Working directory : $PWD

# Start the calculation (safer to use srun)
echo "running script: $1"
echo "args scripts: ${@:2}"

srun python3 $1 ${@:2}  slurm_params.working_directory=$LOCAL_WORK_DIR slurm_params.job_id=$SLURM_JOB_ID slurm_params.slurm_user=$USER
