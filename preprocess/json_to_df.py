import os
import pickle
import json
import argparse

from tqdm import tqdm
import pandas as pd


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--load_fn',
        required=True,
        help="Directory contains original data."
    )
    p.add_argument(
        '--save_fn',
        required=True,
        help="File name to save the dataframe without file format. File format will be added according to args."
    )
    p.add_argument(
        '--return_tsv',
        action='store_true',
        help="If not true, only pickle file will be saved."
    )

    config = p.parse_args()

    return config


def json_to_tsv(file: str):
    cols = ['sentence_id', 'sentence', 'ne']
    df = pd.DataFrame(columns=cols)
    id = 0

    with open(file) as f:
        DATA = json.loads(f.read())

    ne = []
    for document in tqdm(DATA['document']):
        for sentence in document['sentence']:
            df.loc[id, 'sentence_id'] = sentence['id']
            df.loc[id, 'sentence'] = sentence['form']
            labels = dict()
            for entity in sentence['NE']:
                key = entity['id']
                entity.pop('id')
                labels[key] = entity
            ne.append(labels)
            id += 1
    df['ne'] = ne
    
    return df


def main(config):
    filepath = config.load_fn
    save_fn = config.save_fn

    if os.path.isdir(filepath):
        dfs = []
        for file in tqdm(os.listdir(filepath)):
            df = json_to_tsv(os.path.join(filepath, file))
            dfs.append(df)
        data = pd.concat(dfs)
    else:
        data = json_to_tsv(filepath)

    with open(save_fn+'.pickle', "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if config.return_tsv:
        data.to_csv(save_fn+'.tsv', sep='\t', index=False)

if __name__ == '__main__':
    config = define_argparser()
    main(config)