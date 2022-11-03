import os
import shutil

import numpy as np
import pandas as pd
import wandb
import albumentations as A
import ttach as tta
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations.pytorch.transforms import ToTensorV2
from transformers import get_cosine_schedule_with_warmup

from utils.custom_dataset import ImageQuarterDataset, ImageDataset
from utils.custom_model import ImageModel


class Trainer:

    def __init__(self, args, logger, df, splits=None):
        self.args = args
        self.logger = logger

        self.train_transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            A.OneOf([
                A.Transpose(p=1),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
            ], p=0.75),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                        always_apply=False, p=1.0),
            ToTensorV2(),
        ])
        self.valid_transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                        always_apply=False, p=1.0),
            ToTensorV2(),
        ])

        # load dataset setting
        self._make_datasets(splits, df, is_train=args.is_train)

        self.model = self._get_model()
        self.model.to(args.device)

        self.supervised_loss = self._get_loss()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        self.best_valid_f1_score = 0
        self.best_model_folder = None

    def _make_datasets(self, splits, df, is_train):
        if is_train:
            train_idx, valid_idx = splits
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]
            nSamples = sorted(train_df.artist.value_counts())
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            self.class_weights = torch.FloatTensor(normedWeights).to(self.args.device)
            train_df = pd.concat([train_df] * 3)
            valid_df = pd.concat([valid_df] * 3)

            if self.args.method == 'image':
                train_dataset = ImageDataset(self.args, train_df, self.train_transform)
                valid_dataset = ImageQuarterDataset(self.args, valid_df, self.valid_transform)
            elif self.args.method == 'iamge_step':
                raise NotImplementedError('args.method 를 잘 선택 해주세요.')
            else:
                raise NotImplementedError('args.method 를 잘 선택 해주세요.')
            self.train_loader = train_dataset.loader
            self.valid_loader = valid_dataset.loader

            self.step_per_epoch = len(self.train_loader)
        else:
            if self.args.method == 'image':
                test_dataset = ImageDataset(self.args, df, self.valid_transform, is_test=True)
            else:
                raise NotImplementedError('args.method 를 잘 선택 해주세요.')
            self.test_loader = test_dataset.loader

    def train_epoch(self, epoch):
        self.train_loss.reset()
        self.train_acc.reset()

        self.model.train()

        self.optimizer.zero_grad()

        train_iterator = tqdm(self.train_loader, desc='Train Iteration')
        for step, batch in enumerate(train_iterator):
            batch = self.batch_to_device(batch)

            total_step = epoch * self.step_per_epoch + step

            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                logits = self.model(batch['image'])
                loss = self.supervised_loss(logits, batch['label'])
                preds = torch.argmax(logits, dim=-1)

                self.scaler.scale(loss).backward()

                acc = accuracy_score(batch['label'].cpu(), preds.cpu())

                if (total_step + 1) % self.args.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if self.args.scheduler == 'get_cosine_schedule_with_warmup':
                    self.scheduler.step()

                self.train_loss.update(loss.item(), self.args.train_batch_size)
                self.train_acc.update(acc, self.args.train_batch_size)

                if total_step != 0 and total_step % (self.args.eval_steps * self.args.accumulation_steps) == 0:
                    valid_acc, valid_f1_score, valid_loss = self.validate()

                    if self.args.scheduler == 'ReduceLROnPlateau':
                        self.scheduler.step(valid_acc)

                    self.model.train()
                    last_lr = self.scheduler.optimizer.param_groups[0]['lr']
                    if self.args.wandb:
                        wandb.log({
                            'train/loss': self.train_loss.avg,
                            'train/acc': self.train_acc.avg,
                            'train/learning_rate': last_lr,
                            'eval/loss': valid_loss,
                            'eval/acc': valid_acc,
                            'eval/f1_score': valid_f1_score,
                        })

                    self.logger.info(
                        f'STEP {total_step} | eval loss: {valid_loss:.4f} | eval acc: {valid_acc:.4f} | eval f1_score: {valid_f1_score:.4f}'
                    )
                    self.logger.info(
                        f'STEP {total_step} | train loss: {self.train_loss.avg:.4f} | train acc: {self.train_acc.avg:.4f} | lr: {last_lr}'
                    )

                    if valid_f1_score > self.best_valid_f1_score:
                        self.logger.info(f'BEST_BEFORE : {self.best_valid_f1_score:.4f}, NOW : {valid_f1_score:.4f}')
                        self.logger.info('Saving Model...')
                        self.best_valid_f1_score = valid_f1_score
                        self.save_model(total_step)

    def validate(self):
        self.model.eval()

        valid_iterator = tqdm(self.valid_loader, desc="Valid Iteration")

        valid_acc = AverageMeter()
        valid_loss = AverageMeter()

        preds_list = []
        label_list = []
        with torch.no_grad():
            for step, batch in enumerate(valid_iterator):
                batch = self.batch_to_device(batch)

                logits = self.model(batch['image'])
                loss = self.supervised_loss(logits, batch['label'])
                preds = torch.argmax(logits, dim=-1)

                preds_list.append(preds.detach().cpu().numpy())
                label_list.append(batch['label'].detach().cpu().numpy())
                acc = accuracy_score(batch['label'].cpu(), preds.cpu())

                valid_loss.update(loss.item(), self.args.valid_batch_size)
                valid_acc.update(acc, self.args.valid_batch_size)

        preds_list = np.hstack(preds_list)
        label_list = np.hstack(label_list)
        valid_f1_score = f1_score(label_list, preds_list, average='macro')

        return valid_acc.avg, valid_f1_score, valid_loss.avg

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        if self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        else:
            raise NotImplementedError('args.optimizer 를 잘 선택 해주세요.')

        return optimizer

    def _get_scheduler(self):
        if self.args.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=self.args.patience, factor=0.9)
        elif self.args.scheduler == 'get_cosine_schedule_with_warmup':
            total_steps = self.step_per_epoch * self.args.epochs
            warmup_steps = self.step_per_epoch * self.args.warmup_ratio
            scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)
        else:
            raise NotImplementedError('args.scheduler 를 잘 선택 해주세요.')
        return scheduler

    def save_model(self, step):
        if self.best_model_folder:
            shutil.rmtree(self.best_model_folder)

        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path, exist_ok=True)

        file_name = f'FOLD{self.args.fold}_STEP_{step}_F1{self.best_valid_f1_score:.4f}'
        output_path = os.path.join(self.args.output_path, file_name)

        os.makedirs(output_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(output_path, 'model_state_dict.pt'))

        self.logger.info(f'Model Saved at {output_path}')
        self.best_model_folder = output_path

        if self.args.wandb:
            wandb.log({'eval/best_f1_score': self.best_valid_f1_score})

    def batch_to_device(self, batch):
        batch = {k: v.to(self.args.device) for k, v in batch.items()}
        return batch

    def predict(self):
        model_state_dict = torch.load(os.path.join(self.args.saved_model_path, 'model_state_dict.pt'),
                                      map_location=self.args.device)
        self.model.load_state_dict(model_state_dict)

        if self.args.tta:
            transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 90]),
                    # tta.Scale(scales=[1, 2]),
                    # tta.FiveCrops(384, 384),
                    tta.Multiply(factors=[0.7, 1]),
                ]
            )
            tta_model = tta.ClassificationTTAWrapper(self.model, transforms, merge_mode='sum').to(self.args.device)

        test_iterator = tqdm(self.test_loader, desc='Test Iteration')

        preds_list = []
        probs_list = []
        with torch.no_grad():
            for step, batch in enumerate(test_iterator):
                batch = self.batch_to_device(batch)

                if self.args.tta:
                    logits = self.model(batch['image']) + tta_model(batch['image'])
                else:
                    logits = self.model(batch['image'])
                probs_list.append(logits.detach().cpu().numpy())
                preds = torch.argmax(logits, dim=-1)
                preds_list.append(preds.detach().cpu().numpy())

        preds_list = np.hstack(preds_list)
        probs_list = np.vstack(probs_list)

        return preds_list, probs_list

    def _get_loss(self):
        if self.args.loss == 'CrossEntropy':
            loss = nn.CrossEntropyLoss()
        elif self.args.loss == 'WeightedCrossEntropy':
            loss = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            raise NotImplementedError('args.loss 를 잘 선택 해주세요.')
        return loss

    def _get_model(self):
        if self.args.is_train:
            if self.args.method == 'multimodal':
                raise NotImplementedError('아직 미완성')
            elif self.args.method == 'image':
                model = ImageModel(self.args)
            else:
                raise NotImplementedError('args.method 를 잘 선택 해주세요.')
        else:
            if 'multimodal' in self.args.saved_model_path:
                raise NotImplementedError('아직 미완성')
            elif 'resnext50_32x4d' in self.args.saved_model_path:
                self.args.model_name_or_path = 'resnext50_32x4d'
                model = ImageModel(self.args)
            elif 'vit_base_patch16_384' in self.args.saved_model_path:
                self.args.model_name_or_path = 'vit_base_patch16_384'
                model = ImageModel(self.args)
            elif 'resnet50' in self.args.saved_model_path:
                self.args.model_name_or_path = 'resnet50'
                model = ImageModel(self.args)
            else:
                raise NotImplementedError('좀 더 고민해봐....... 에러처리 더 해야 할 듯')

        return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
