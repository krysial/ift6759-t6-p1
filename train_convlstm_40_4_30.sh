#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o ./results/slurm-%A_%a.out # STDOUT

module load imkl/2018.3.222
module load openmpi/3.1.2
module load cuda/10.0.130
module load python/3.7
source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/10.0.130/extras/CUPTI/lib64
python train.py data/admin_cfg.json -u data/user_cfg.json --model=convlstm  --input_past_interval=30 --crop-size=40 --seq-len=4 --batch-size=32 --epoch=150 --real -lr=0.1