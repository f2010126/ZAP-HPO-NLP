#!/bin/bash
#SBATCH -o /work/dlclarge1/dsengupt-zap_hpo_og/logs/nlp.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-zap_hpo_og/logs/nlp.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J NLPPipeline
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

deactivate
echo "Activate environment"
source ~/nlp_env/bin/activate
cd $(ws_find zap_hpo_og)/zap_nlp/huggingface_trials/
GPU=$(nvidia-smi  -L | wc -l)
: '
accelerate launch --multi_gpu --gpu_ids "all" --num_processes $GPU trainer_gpu.py --model_name "bert-base-german-cased" --dataset_name "amazon_reviews_multi" --exp_name "test-pipeline" --job_name "4GPU"
accelerate launch --multi_gpu --gpu_ids "all" --num_processes $GPU trainer_gpu.py --num_epochs 5 --batch_size 16 --lr 2e-05 --seed 42 --model_name "distilbert-base-cased" --dataset_name "amazon_reviews_multi" --split-lang 'en' --exp_name "test-pipeline" --group_name 'DistilBERT' --job_name "4GPU"
'
accelerate launch --multi_gpu --gpu_ids "all" --num_processes $GPU trainer_gpu.py --num_epochs 5 --batch_size 16 --lr 2e-05 --seed 42 --model_name "distilbert-base-cased" --dataset_name "amazon_reviews_multi" --split-lang 'en' --exp_name "test-pipeline" --group_name 'DistilBERT' --job_name "4GPU"
echo "Deactivate environment and exit"
deactivate
