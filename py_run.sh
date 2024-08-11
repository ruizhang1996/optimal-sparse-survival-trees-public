#!/bin/bash
#SBATCH -o sbatchout/osst2.out
#SBATCH --mem=100G
#SBATCH --job-name=osst2
valgrind --leak-check=full --show-leak-kinds=all --log-file=valgrind-output.txt python3 osst/example.py