# Image baseline Quarter size image resnext50_32x4d Train
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_resnext50_32x4d
#python inference.py --device 1 --output_path_list ./saved_model/image_resnext50_32x4d --predict_path ./predict/image_resnext50_32x4d

# Image baseline full size image resnext50_32x4d Train
#python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_full_size_resnext50_32x4d
#python inference.py --device 1 --output_path_list ./saved_model/image_full_size_resnext50_32x4d --predict_path ./predict/image_full_size_resnext50_32x4d

# Image baseline full size image resnext50_32x4d Train Validation Quarter size
python train.py --is_train --use_amp --device 1 --epochs 30 --train_batch_size 32 --wandb --model_name_or_path resnext50_32x4d --output_path ./saved_model/image_full_size_valid_quarter_resnext50_32x4d
python inference.py --device 1 --output_path_list ./saved_model/image_full_size_valid_quarter_resnext50_32x4d --predict_path ./predict/image_full_size_valid_quarter_resnext50_32x4d
