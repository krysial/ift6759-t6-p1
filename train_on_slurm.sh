#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
#SBATCH -o ./results/slurm-%A_%a.out # STDOUT

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/ift6759-env
source $SLURM_TMPDIR/ift6759-env/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt