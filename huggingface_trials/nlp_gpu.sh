#!/bin/bash
#SBATCH -o /work/dlclarge1/dsengupt-zap_hpo_og/logs/nlp.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-zap_hpo_og/logs/nlp.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J Feat_fast
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

deactivate
echo "Activate environment"
source ~/nlp_env/bin/activate
cd $(ws_find zap_hpo_og)/zap_nlp/huggingface_trials/
accelerate launch trainer_gpu.py
echo "Deactivate environment and exit"
deactivate
