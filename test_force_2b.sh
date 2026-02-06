#!/bin/sh
#SBATCH --job-name=binary_avg_test_Tang_2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lottc@ufl.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=4
#SBATCH --mem=24gb
#SBATCH --time=00-02:00:00
#SBATCH --output=Testing_Tang_no_std_Results.txt

export PATH=/blue/bala1s/lottc/envs/Pytorch_ML_env/bin:$PATH

#module load intel/2025.1.0
#module load pytorch/1.13

for i in $(seq 1 1 48)
do
	python3 testing_force_2b.py --fl_no $i --num_lyrs 6 --lyr_wdt 250 --num_neig 26
done
