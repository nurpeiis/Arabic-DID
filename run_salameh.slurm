#!/bin/bash
#SBATCH --job-name=salameh
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --mail-user=nurpeiis@nyu.edu
#SBATCH --gres=gpu:1


SRCDIR=$SCRATCH/

cd $SRCDIR/arabic-did/salameh
source $SRCDIR/miniconda3/bin/activate
conda activate mllu
python salameh