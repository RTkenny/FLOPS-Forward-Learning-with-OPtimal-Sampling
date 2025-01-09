CUDA_VISIBLE_DEVICES=3 accelerate launch train_cnn.py \
    --is_log=True \
    --output_dir='./output/mlp' \
    --sample_n=100 \
    --per_device_train_batch_size=256 \
    --dataset='cifar10' \
    --gradient_estimation_strategy='perturb_outside_tensor_wise' \
    --log_with='wandb' \

