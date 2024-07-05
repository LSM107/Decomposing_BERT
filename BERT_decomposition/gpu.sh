#! /bin/bash
#SBATCH --nodes=51
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=14
#SBATCH -o ./_out/%j.%N.out
#SBATCH -e ./_err/%j.%N.err

#=============================================

echo "start at:" `date`
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"


# (1) print your name
# (2) excute test.py python file
python pruning_accuracy_1.py
lsb_release -a