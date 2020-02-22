#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -o ./results/slurm-%A_%a.out # STDOUT

start_indices=(0    200 400 600 800  1000 1200 1400 1600 1800 2000)
end_indices=(  200  400 600 800 1000 1200 1400 1600 1800 2000)

module load python/3.7
source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate

if [ "${#end_indices[@]}" -gt "$(($SLURM_ARRAY_TASK_ID))" ]
then
    echo "start ${start_indices[$SLURM_ARRAY_TASK_ID]}"
    echo "start ${end_indices[$SLURM_ARRAY_TASK_ID]}"
    python preprocessing.py -u --start-index=${start_indices[$SLURM_ARRAY_TASK_ID]} --end-index=${end_indices[$SLURM_ARRAY_TASK_ID]}
elif [ "${#end_indices[@]}" -eq "$(($SLURM_ARRAY_TASK_ID))" ]
then
    echo "start ${start_indices[$SLURM_ARRAY_TASK_ID]}"
    python preprocessing.py -u --start-index=${start_indices[$SLURM_ARRAY_TASK_ID]}
else
    echo "OUT OF BOUNDS"
fi