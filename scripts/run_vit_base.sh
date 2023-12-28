cd ../trainers/
runName="hypermae_vit_base_v1"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_pretrain_log.py  --batch_size 256 \
    --world_size 1 \
    --accum_iter 2 \
    --model hyper_mae_vit_base \
    --engine pretrain \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ./data/imagenet/ \
    --dist_url tcp://localhost:10009 \
    --output_dir ./save/${runName} \
    --log_dir ./save/${runName} \
    --run_name ${runName} \
    --log_to_wandb 1 \
    --norm_pix_loss \
    --vis 1 \
    --dist_backend nccl \

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_log.py \
    --accum_iter 2 \
    --batch_size 128 \
    --engine finetune \
    --model vit_base_patch16 \
    --finetune ./save/${runName}/checkpoint-399.pth \
    --dist_url tcp://localhost:10007 \
    --data_path ./data/imagenet/ \
    --output_dir ./save/${runName}_ft/ \
    --log_dir ./save/${runName}_ft/ \
    --nb_classes 1000 \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --run_name ${runName}_ft \
    --log_to_wandb 1 \
    --dist_backend nccl \