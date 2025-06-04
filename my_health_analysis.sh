#!/bin/bash
#SBATCH --job-name=my_first_health_analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/health_analysis_%j.out

echo "Starting health data analysis on $(hostname)"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Load modules (if on Wynton)
# module load python/3.9

# Run your parallel analysis
python parallel_patient_analysis.py
echo "Analysis complete!"
