#!/bin/bash

# Job Name
#SBATCH --job-name=mpi-force2vec
# Number of Nodes
#SBATCH --nodes=3
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=24

module load mpi/mpich/latest

srun --mpi=pmi2 ./bin/Force2Vec -input ./datasets/input/cora.mtx -output ./datasets/output/ -iter 5 -batch 192 -threads 32 -option 5 -bs 1 -dim 2
