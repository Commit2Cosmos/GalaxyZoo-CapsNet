#$ -S /bin/bash
#$ -q serial
#$ -N test

source /etc/profile

module add anaconda3
module add cuda

python /mmfs1/home/users/belov/Data/create_paths_votes.py