import os
import pickle

import numpy as np
import pandas as pd

from glob import glob

from utils.data import get_df
from utils.setting import Setting
from utils.trainer import Trainer


def main():
    args, logger = Setting().run()

    df = get_df(args)

    if args.output_path_list:
        model_list = []
        for output_path in args.output_path_list:
            model_list += glob(os.path.join(output_path, '*'))
    else:
        model_list = glob(os.path.join(args.output_path, '*'))
    model_list = sorted(model_list)

    if not model_list:
        logger.info('저장된 모델 없음....종료!')
        return

    if args.ensemble and os.path.exists('./predict/ensemble_output_probs.pkl'):
        output_probs = load_pickle('./predict/ensemble_output_probs.pkl')
    else:
        output_probs = np.zeros((df.shape[0], args.num_labels))
    for i, model_name in enumerate(model_list, start=1):
        args.saved_model_path = model_name
        logger.info(f'{i} 번째 predict 진행 중!')
        trainer = Trainer(args, logger, df)
        preds_list, probs_list = trainer.predict()

        output_probs += probs_list

    pred_answer = np.argmax(output_probs, axis=-1).tolist()

    output_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/sample_submission.csv'))
    output_df['artist'] = pred_answer
    output_df['artist'] = output_df['artist'].map(args.idx_to_label)

    if not os.path.exists(args.predict_path):
        os.makedirs(args.predict_path, exist_ok=True)

    file_save_path = os.path.join(args.predict_path, f'submission_{args.predict_path.split("/")[-1]}')
    output_df.to_csv(f'{file_save_path}.csv', index=False)
    logger.info(f'File Save at {file_save_path}')

    if args.ensemble:
        output_probs_save_path = './predict/ensemble_output_probs.pkl'
        save_pickle(output_probs_save_path, output_probs)
    else:
        output_probs_save_path = os.path.join(args.predict_path, 'output_probs.pkl')
        save_pickle(output_probs_save_path, output_probs)
    logger.info(f'File Save at {output_probs_save_path}')


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, x):
    with open(path, 'wb') as f:
        pickle.dump(x, f)




if __name__ == '__main__':
    main()
