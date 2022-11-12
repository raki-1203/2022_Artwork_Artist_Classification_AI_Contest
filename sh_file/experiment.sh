# Image baseline Quarter size image resnext50_32x4d Train -> 0.301354
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_resnext50_32x4d
#python inference.py --device 1 --output_path_list ./saved_model/image_resnext50_32x4d --predict_path ./predict/image_resnext50_32x4d

# Image baseline full size image resnext50_32x4d Train -> 0.558937
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_full_size_resnext50_32x4d
#python inference.py --device 1 --output_path_list ./saved_model/image_full_size_resnext50_32x4d --predict_path ./predict/image_full_size_resnext50_32x4d

# Image baseline full size image resnext50_32x4d Train Validation Quarter size -> 0.714959
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_full_size_valid_quarter_resnext50_32x4d
#python inference.py --device 1 --output_path_list ./saved_model/image_full_size_valid_quarter_resnext50_32x4d --predict_path ./predict/image_full_size_valid_quarter_resnext50_32x4d

# Image baseline Train full size & Quarter size image resnext50_32x4d Train Validation Quarter size -> 0.748545
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_train_fq_valid_q_resnext50_32x4d --scheduler get_cosine_schedule_with_warmup --loss WeightedCrossEntropy
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_resnext50_32x4d --predict_path ./predict/image_train_fq_valid_q_resnext50_32x4d

# Image baseline Train full size & Quarter size image vit_base_patch16_384 Train Validation Quarter size -> 0.633569, 0.589128
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 8 --accumulation_steps 4 --wandb --model_name_or_path vit_base_patch16_384 --output_path ./saved_model/image_train_fq_valid_q_vit_base_patch16_384 --scheduler get_cosine_schedule_with_warmup --loss WeightedCrossEntropy
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_vit_base_patch16_384 --predict_path ./predict/image_train_fq_valid_q_vit_base_patch16_384
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_vit_base_patch16_384 --predict_path ./predict/image_train_fq_valid_q_vit_base_patch16_384 --tta

# Image baseline Train full size & Quarter size image vit_base_patch16_384 Train Validation Quarter size -> 0.724353, 0.696572
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 8 --accumulation_steps 4 --wandb --model_name_or_path vit_base_patch18_384 --output_path ./saved_model/image_train_fq_valid_q_lr5e-5_ReduceLR_vit_base_patch16_384 --loss WeightedCrossEntropy --lr 5e-5
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_lr5e-5_ReduceLR_vit_base_patch16_384 --predict_path ./predict/image_train_fq_valid_q_lr5e-5_ReduceLR_vit_base_patch16_384
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_lr5e-5_ReduceLR_vit_base_patch16_384 --predict_path ./predict/image_train_fq_valid_q_lr5e-5_ReduceLR_vit_base_patch16_384 --tta

# Image baseline Train full size & Quarter size image vit_large_patch16_384 Train Validation Quarter size
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 2 --accumulation_steps 16 --wandb --lr 8e-5 --model_name_or_path vit_large_patch16_384 --output_path ./saved_model/image_train_fq_valid_q_vit_large_patch16_384 --loss WeightedCrossEntropy
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_vit_large_patch16_384 --predict_path ./predict/image_train_fq_valid_q_vit_large_patch16_384

# Image baseline Train full size & Quarter size image Resnet50 Train Validation Quarter size -> 0.672170
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --lr 1e-3 --model_name_or_path resnet50 --output_path ./saved_model/image_train_fq_valid_q_resnet50 --loss WeightedCrossEntropy
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_resnet50 --predict_path ./predict/image_train_fq_valid_q_resnet50 --tta

# Image baseline Train full size & Quarter size image resnext50_32x4d Train Validation Quarter size cutmix 0.5 beta 1 -> 0.742406
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_train_fq_valid_q_cutmix0.5_resnext50_32x4d --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_cutmix0.5_resnext50_32x4d --predict_path ./predict/image_train_fq_valid_q_cutmix0.5_resnext50_32x4d --tta

# Image baseline Train full size & Quarter size image resnext50_32x4d Train Validation Quarter size cutmix 0.5 beta 1 scheduler factor 0.5-> 0.754724
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_train_fq_valid_q_cutmix0.5_scheduler0.5_resnext50_32x4d --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5
#python inference.py --device 1 --output_path_list ./saved_model/image_train_fq_valid_q_cutmix0.5_scheduler0.5_resnext50_32x4d --predict_path ./predict/image_train_fq_valid_q_cutmix0.5_scheduler0.5_resnext50_32x4d --tta

# Image baseline -> 0.79893
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_baseline_resnext50_32x4d --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_resnext50_32x4d --predict_path ./predict/image_baseline_resnext50_32x4d --tta

# Image baseline + validation dataset 랜덤하게 crop 되도록 계속 변경되도록 만듦 -> 0.7912232973
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_baseline_validation_randomcrop_resnext50_32x4d --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_validation_randomcrop_resnext50_32x4d --predict_path ./predict/image_baseline_validation_randomcrop_resnext50_32x4d --tta

# Image baseline + model efficientnet_b4 -> 0.813227
#python train.py --is_train --use_amp --device 0 --epochs 30 --train_batch_size 16 --accumulation_steps 2 --wandb --model_name_or_path efficientnet_b4 --output_path ./saved_model/image_baseline_efficientnet_b4 --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_efficientnet_b4 --predict_path ./predict/image_baseline_efficientnet_b4 --tta

# Image baseline + model efficientnet_b4 Cross Validation -> 0.84400
#python train.py --is_train --use_amp --device 0 --epochs 30 --train_batch_size 16 --accumulation_steps 2 --wandb --model_name_or_path efficientnet_b4 --output_path ./saved_model/image_baseline_efficientnet_b4 --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5 --cv
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_efficientnet_b4 --predict_path ./predict/image_baseline_efficientnet_b4_cv --tta

# Image baseline + model resnext50_32x4d Cross Validation -> 0.830019
#python train.py --is_train --use_amp --device 0 --epochs 30 --train_batch_size 32 --accumulation_steps 1 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_baseline_resnext50_32x4d --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5 --cv
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_resnext50_32x4d --predict_path ./predict/image_baseline_resnext50_32x4d_cv --tta

# Image baseline + model efficientnet_b4, resnext50_32x4d Ensemble Inference -> 0.860458
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_efficientnet_b4 ./saved_model/image_baseline_resnext50_32x4d --predict_path ./predict/image_baseline_ensemble_efficientnet_b4_resnext50_32x4d_cv --tta

# Image baseline + model convnext_tiny_384_in22ft1k Cross Validation -> 0.83268, 0.870126
#python train.py --is_train --use_amp --device 1 --epochs 20 --train_batch_size 32 --accumulation_steps 1 --wandb --model_name_or_path convnext_tiny_384_in22ft1k --output_path ./saved_model/image_baseline_convnext_tiny_384_in22ft1k --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5 --cv
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_convnext_tiny_384_in22ft1k --predict_path ./predict/image_baseline_convnext_tiny_384_in22ft1k_cv --tta
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_efficientnet_b4 ./saved_model/image_baseline_resnext50_32x4d ./saved_model/image_baseline_convnext_tiny_384_in22ft1k --predict_path ./predict/image_baseline_ensemble_efficientnet_b4_resnext50_32x4d_convnext_tiny_384_in22ft1k_cv --tta --ensemble

# Image baseline + cutout + model efficientnet_b4 Cross Validation -> 0.8242523, 0.869911
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 16 --accumulation_steps 2 --wandb --model_name_or_path efficientnet_b4 --output_path ./saved_model/image_baseline_cutout_efficientnet_b4 --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5 --cv
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_cutout_efficientnet_b4 --predict_path ./predict/image_baseline_cutout_efficientnet_b4_cv --tta
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_efficientnet_b4 ./saved_model/image_baseline_resnext50_32x4d ./saved_model/image_baseline_convnext_tiny_384_in22ft1k ./saved_model/image_baseline_cutout_efficientnet_b4 --predict_path ./predict/image_baseline_ensemble_efficientnet_b4_resnext50_32x4d_convnext_tiny_384_in22ft1k_cutout_efficientnet_b4_cv --tta --ensemble

# Image baseline + model res2net50_26w_4s Cross Validation -> 0.829134, 0.872181
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --accumulation_steps 1 --wandb --model_name_or_path res2net50_26w_4s --output_path ./saved_model/image_baseline_res2net50_26w_4s --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5 --cv
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_res2net50_26w_4s --predict_path ./predict/image_baseline_res2net50_26w_4s_cv --tta
#python inference.py --device 1 --output_path_list ./saved_model/image_baseline_res2net50_26w_4s ./saved_model/image_baseline_efficientnet_b4 ./saved_model/image_baseline_resnext50_32x4d ./saved_model/image_baseline_convnext_tiny_384_in22ft1k --predict_path ./predict/image_baseline_ensemble_efficientnet_b4_resnext50_32x4d_convnext_tiny_384_in22ft1k_res2net50_26w_4s --tta --ensemble

# Image baseline + model vit_base_patch16_384 Cross Validation ->
python train.py --is_train --use_amp --device 0 --epochs 30 --train_batch_size 8 --accumulation_steps 4 --wandb --model_name_or_path vit_base_patch16_384 --output_path ./saved_model/image_baseline_vit_base_patch16_384 --loss WeightedCrossEntropy --beta 1 --cutmix_prob 0.5 --cv --lr 5e-5
python inference.py --device 1 --output_path_list ./saved_model/image_baseline_vit_base_patch16_384 --predict_path ./predict/image_baseline_vit_base_patch16_384_cv --tta --img_size 224
python inference.py --device 1 --output_path_list ./saved_model/image_baseline_vit_base_patch16_384 ./saved_model/image_baseline_res2net50_26w_4s ./saved_model/image_baseline_efficientnet_b4 ./saved_model/image_baseline_resnext50_32x4d ./saved_model/image_baseline_convnext_tiny_384_in22ft1k --predict_path ./predict/image_baseline_ensemble_efficientnet_b4_resnext50_32x4d_convnext_tiny_384_in22ft1k_res2net50_26w_4s_vit_base_patch16_384 --tta --ensemble