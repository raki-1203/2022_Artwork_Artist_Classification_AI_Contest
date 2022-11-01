import os

import pandas as pd


def get_df(args):
    if args.is_train:
        df = pd.read_csv(args.path_to_train_data)
        df['img_path'] = df['img_path'].apply(lambda x: os.path.join(args.image_path, '/'.join(x.split('/')[-2:])))
        df['artist'] = df['artist'].map(args.label_to_idx)
    else:
        df = pd.read_csv(args.path_to_test_data)
        df['img_path'] = df['img_path'].apply(lambda x: os.path.join(args.image_path, '/'.join(x.split('/')[-2:])))
    return df
