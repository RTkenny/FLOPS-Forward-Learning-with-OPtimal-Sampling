# perturb_outside_all_params
CUDA_VISIBLE_DEVICES=0 accelerate launch train_mlp.py \
    --is_log=True \
    --output_dir='./output/mlp' \
    --sample_n=100 \
    --gradient_estimation_strategy='perturb_outside_all_params' \
    --log_with='wandb' \
    #--seed=50 \