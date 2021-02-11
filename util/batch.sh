#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal

#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
#SBATCH --output=xor_%J_stdout.txt
#SBATCH --error=xor_%J_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=xor_test
#SBATCH --mail-user=andrewhfagg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/fagg/aml/demos/basics
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate
# Change this line to start an instance of your experiment
python xor_base.py

