import cv2
import torch
import albumentations as A

from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):

    def __init__(self, args, df, transforms, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.collate_fn = CollateImage(is_test)
        self.loader = DataLoader(dataset=self,
                                 batch_size=args.train_batch_size if not is_test else args.valid_batch_size,
                                 shuffle=True if not is_test else False,
                                 sampler=None,
                                 collate_fn=self.collate_fn)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_path = self.df['img_path'].iloc[idx]
        image = cv2.imread(img_path)

        h, w, c = image.shape
        new_h = h // 4
        new_w = w // 4
        image = A.RandomCrop(new_h, new_w)(image=image)['image']

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        # Label
        if self.is_test:
            return {'image': image}
        else:
            label = self.df['artist'].iloc[idx]
            return {'image': image, 'label': label}


class CollateImage:

    def __init__(self, is_test):
        self.is_test = is_test

    def __call__(self, batches):
        b_input_images = []
        if not self.is_test:
            b_labels = []

        for b in batches:
            b_input_images.append(b['image'])

            if not self.is_test:
                b_labels.append(b['label'])

        t_input_images = torch.stack(b_input_images)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'image': t_input_images}
        else:
            t_labels = torch.tensor(b_labels)  # List -> Tensor
            return {'image': t_input_images, 'label': t_labels}
