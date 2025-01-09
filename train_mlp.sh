# different gradient estimation strategies:
# 'lr', 'allocater', 'vanilla_lr', 'tensor_wise_perturb', 'tensor_wise_perturb_reuse', 'perturb_outside_all_params', 'perturb_outside_tensor_wise', 'non-diff', 'bp'
CUDA_VISIBLE_DEVICES=0 accelerate launch train_mlp.py \
    --is_log=True \
    --output_dir='./output/mlp' \
    --sample_n=9 \
    --gradient_estimation_strategy='perturb_outside_all_params_allocator' \
    --log_with='wandb' \
    #--seed=50 \