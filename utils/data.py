import os

import pandas as pd


def get_df(args):
    if args.is_train:
        df = pd.read_csv(args.path_to_train_data)
        # 잘못 레이블링 된 데이터 수정
        df.loc[3896] = {'id': 3896, 'img_path': './train/3896.jpg', 'artist': 'Titian'}
        df.loc[3986] = {'id': 3986, 'img_path': './train/3986.jpg', 'artist': 'Alfred Sisley'}
        df['img_path'] = df['img_path'].apply(lambda x: os.path.join(args.image_path, '/'.join(x.split('/')[-2:])))
        df['artist'] = df['artist'].map(args.label_to_idx)
    else:
        df = pd.read_csv(args.path_to_test_data)
        df['img_path'] = df['img_path'].apply(lambda x: os.path.join(args.image_path, '/'.join(x.split('/')[-2:])))
    return df
