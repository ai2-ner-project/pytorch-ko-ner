import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--load_path',
        required=True,
        help="Path to load data file."
    )
    p.add_argument(
        "--save_path",
        required=True,
        help="Path to save preprocessed dataset."
    )
    p.add_argument(
        '--test_size',
        required=True,
        default=.2,
        type=float,
        help="Ratio of test data over dataset. Float only acceptable."
    )

    p.add_argument(
        "--pass_drop_o",
        action="store_true",
        help="If true, train/test set contains O sentences without any NE."
    )
    p.add_argument(
        "--test_o_size",
        default=0,
        type=float,
        help="Ratio of sentences without any NE over test data. It only work when pass_drop_o is False."
    )

    p.add_argument(
        '--save_all',
        action="store_true",
        help="If true, save not splited data."
    )
    p.add_argument(
        "--return_tsv",
        action="store_true",
        help="If not true, only pickle files will be saved."
    )

    config = p.parse_args()

    return config


def main(config):
    """
    Prepreccs data files and split as train and test set.
    1. Read all data files in the directory.
    2. Concatenate files and drop useless columns.
    3. Add labels for experiments and data split.
    """
    load_path = config.load_path
    save_path = config.save_path

    # Read data files.
    if os.path.isdir(load_path):
        file_list = [fn for fn in os.listdir(load_path) if fn.endswith("pickle")]
    print(f"{len(file_list)} files found : ", file_list)

    for i, file in enumerate(file_list):
        load_file = os.path.join(load_path, file)
        file_list[i] = pd.read_pickle(load_file)
        print(f"file {i} : ", file_list[i].shape[0])

    # Concatenate all data files in the directory.
    data = pd.concat(file_list, axis=0, ignore_index=True)
    print(f"|data before preprocessing| {data.shape[0]}")

    # Add source of sentence
    # S: Conversation / N: News
    data['source'] = data['sentence_id'].str[0]
    data = data.drop(columns=['sentence_id'], axis=1)
    data = data[data['sentence'].map(len) > 0].reset_index(drop=True)

    NE_list = ["PS", "FD", "TR", "AF", "OG", "LC", "CV", "DT", "TI", "QT", "EV", "AM", "PT", "MT", "TM"]
    NE_counter = dict(zip(NE_list, [0] * 15))

    def get_label_list(ne_dict):
        label_list = []
        for _, values in ne_dict.items():
            label_list.append(values['label'][:2])

        return label_list

    data['ne_label_list'] = data['ne'].map(get_label_list)
    for label_list in data['ne_label_list']:
        for label in label_list:
            for ne in NE_list:
                if label == ne:
                    NE_counter[ne] += 1

    NE_list_sorted = pd.DataFrame(NE_counter, index = ['count']).T.reset_index().sort_values(by='count', ignore_index=True)

    sentence_class = []
    for label_list in data['ne_label_list']:
        if len(label_list) < 1:
            sentence_class.append('Out')
            continue
        else:
            for ne in NE_list_sorted['index']:
                if ne in label_list:
                    sentence_class.append(ne)
                    break

    data['sentence_class'] = sentence_class
    print(f"|data after preprocessing| {data.shape[0]} / before dropping O sentences")

    train, test = train_test_split(data, test_size=config.test_size, stratify=data['sentence_class'])
    print(f"|train| {train.shape[0]} / |test| {test.shape[0]} / before dropping O sentences")

    # Drop "O" sentences and add some sample for test
    if config.pass_drop_o:
        pass
    else:
        train = train[train['ne_label_list'].map(len) > 0]
        test_o = test[test['ne_label_list'].map(len) == 0]
        test = test[test['ne_label_list'].map(len) > 0]
        if config.test_o_size > 0:
            test_o['n_words'] = test_o['sentence'].map(lambda x: len(x.split()))
            test_o_from_n = test_o[test_o['source'] == 'N']
            test_o_from_s = test_o[test_o['source'] == 'S']

            test_o_ratio = config.test_o_size / (1 - config.test_o_size)
            num_o_sample = int(test.shape[0] * test_o_ratio)
            num_o_from_n = min(num_o_sample // 2, test_o_from_n.shape[0])
            num_o_from_s = num_o_sample - num_o_from_n
            test_o_from_n = test_o_from_n.sample(n=num_o_from_n, random_state=42)
            test_o_from_s = test_o_from_s.sample(n=num_o_from_s, weights='n_words', random_state=42)                
            test = pd.concat([test, test_o_from_n, test_o_from_s])

    data = data.drop(columns=["ne_label_list"])
    train = train.drop(columns=["ne_label_list"])
    test = test.drop(columns=["ne_label_list", "n_words"])
    print(f"|train| {train.shape[0]} / |test| {test.shape[0]} / after dropping and sampling O sentences")

    if config.save_all:
        data.to_pickle(os.path.join(save_path, 'data.pickle'))
    train.to_pickle(os.path.join(save_path, 'train.pickle'))
    test.to_pickle(os.path.join(save_path, 'test.pickle'))
    
    if config.return_tsv:
        if config.save_all:
            data.to_csv(os.path.join(save_path, 'data.tsv'), sep='\t', index=False)
        train.to_csv(os.path.join(save_path, 'train.tsv'), sep='\t', index=False)
        test.to_csv(os.path.join(save_path, 'test.tsv'), sep='\t', index=False)


if __name__ == "__main__":
    config = define_argparser()
    main(config)