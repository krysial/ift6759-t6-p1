#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10000M
#SBATCH -o ./results/slurm-%A_%a.out # STDOUT

module load python/3.7
source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate
python train.py data/admin_cfg.json