#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -o ./results/slurm-%A_%a.out # STDOUT

module load python/3.7
source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate
python preprocessing.py