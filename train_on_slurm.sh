#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10000M
#SBATCH -o ./results/slurm-%A_%a.out # STDOUT

module load imkl/2018.3.222
module load openmpi/3.1.2
module load cuda/10.0.130
module load python/3.7
source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/10.0.130/extras/CUPTI/lib64
python train.py data/admin_cfg.json --model=lrcn --crop-size=40 --seq-len=5 --batch-size=10