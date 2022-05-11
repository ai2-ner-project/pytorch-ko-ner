import pickle
import pandas as pd

def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts


def read_pickle(fn):
    with open(fn, 'rb') as f:
        dataset = pickle.load(f)
    data = pd.DataFrame(dataset.pop('data'))
    return data
