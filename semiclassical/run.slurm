#!/bin/bash
#SBATCH --job-name=semiclassical
#SBATCH --output=semiclassical_%j.out
#SBATCH --error=semiclassical_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00

module load python

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python semiclassical.py input_benchmark.txt
