#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=48
#SBATCH -o ./st09/_out/%j.%N.out
#SBATCH -e ./st09/_err/%j.%N.err
#=============================================

echo "start at:" `date`
echo "node: $HOSTNAME"
echo "jobid: $SLURM_JOB_ID"

# (1) print your name
# (2) excute test.py python file
python file_name.py
lsb_release -a