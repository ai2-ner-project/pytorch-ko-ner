import os
import argparse
import pickle

import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import load_metric
from seqeval.metrics import classification_report
import torch

# huggingface tokenizer/model
from transformers import AutoModelForTokenClassification

# huggingface trainer
from transformers import Trainer
from transformers import TrainingArguments

# Customize encoder
from ner.ner_dataset import NERCollator
from ner.ner_dataset import NERDatasetPreEncoded


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_folder',
                   required=True,
                   help="Directory to save trained model.")
    p.add_argument('--data_fn',
                   required=True,
                   help="Data file name encoded by encoding.py to train the model.")

    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs_per_fold', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=100)

    p.add_argument('--use_kfold', action='store_true')
    p.add_argument('--n_splits', type=int, default=1)

    config = p.parse_args()

    return config


def get_pretrained_model(model_name: str, num_labels: int):
    """
    Basically, use AutoModelForTokenClassification from Huffingface.
    This function remains for future issue.
    """
    model_loader = AutoModelForTokenClassification
    return model_loader.from_pretrained(model_name, num_labels=num_labels)


def load_data(fn, use_kfold=False, n_splits=5, shuffle=True):
    """
    Load tsv data as Dataframe.
    If use_kfold is true, a new column ['fold'] will be added for indexing each fold.
    """
    # Get sentences and labels from a dataframe.
    with open(fn, "rb") as f:
        dataset = pickle.load(f)
    data = pd.DataFrame(dataset.pop('data'))

    if use_kfold:
        skf = StratifiedKFold(
            n_splits=n_splits, random_state=42, shuffle=shuffle)
        data['fold'] = -1
        for n_fold, (_, v_idx) in enumerate(skf.split(data, data['sentence_class'])):
            data.loc[v_idx, 'fold'] = n_fold
        data['id'] = [x for x in range(len(data))]

    return data, dataset


def split_dataset(data, use_kfold=False, n_fold=None, valid_ratio=.2, shuffle=False):
    """
    Split data into train and validation.
    Size of validation set will be determined by 'n_fold' when 'use_kfold' is True, otherwise determined by 'valid_ratio'.
    'shuffle' will affect only in case of 'use_kfold' is False.
    """
    if use_kfold == True:
        train = data[data['fold'] != n_fold]
        valid = data[data['fold'] == n_fold]
    else:
        train, valid = train_test_split(
            data, test_size=valid_ratio, random_state=42, shuffle=shuffle, stratify=data['sentence_class'])

    train_dataset = NERDatasetPreEncoded(train['input_ids'].values, train['attention_mask'].values, train['labels'].values)
    valid_dataset = NERDatasetPreEncoded(valid['input_ids'].values, valid['attention_mask'].values, valid['labels'].values)

    return train_dataset, valid_dataset


class compute_metrics():

    def __init__(self, index_to_label):
        self.index_to_label = index_to_label

    def __call__(self, pred):
        """
        Compute metrics use "seqeval"
        It evaluates based on Entity Level F1 score.
        """
        metric = load_metric('seqeval')

        labels = pred.label_ids
        predictions = pred.predictions.argmax(2)

        # Discard special tokens based on true_labels.
        true_predictions = [[self.index_to_label[p] for p, l in zip(
            prediction, label) if l >= 0] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.index_to_label[l] for p, l in zip(prediction, label) if l >= 0]
                    for prediction, label in zip(predictions, labels)]

        results = metric.compute(
            predictions=true_predictions, references=true_labels)
        eval_results = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        print(classification_report(true_labels, true_predictions))

        return eval_results


def train_one_fold(data, n_fold, data_args, config):
    pretrained_model_name = data_args['pretrained_model_name'].replace('/', '_')

    label_to_index = data_args['label_info']['label_to_index']
    index_to_label = data_args['label_info']['index_to_label']
    pad_token = data_args['pad_token']

    train_dataset, valid_dataset = split_dataset(
        data, use_kfold=config.use_kfold, n_fold=n_fold, valid_ratio=config.valid_ratio, shuffle=True)
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    # Get pretrained model and tokenizer.
    model = get_pretrained_model(
        data_args['pretrained_model_name'], len(label_to_index))

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) /
                             total_batch_size * config.n_epochs_per_fold)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '# of total_iters =', n_total_iterations,
        '# of warmup_iters =', n_warmup_steps,
    )

    training_args = TrainingArguments(
        output_dir=f".checkpoints/{pretrained_model_name}.{n_fold}",
        num_train_epochs=config.n_epochs_per_fold,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs_per_fold,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=NERCollator(pad_token=pad_token,
                                  with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics(index_to_label),
    )

    trainer.train()

    fn_prefix = '.'.join([pretrained_model_name, 
                        f"{config.n_epochs_per_fold}_epochs", 
                        f"{config.max_length}_length",
                        f"{n_fold}_fold", 
                        "pth"])
    model_fn = os.path.join(config.model_folder, fn_prefix)

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': trainer.model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
    }, model_fn)


def main(config):
    data, data_args = load_data(config.data_fn, use_kfold=config.use_kfold,
                     n_splits=config.n_splits, shuffle=True)

    for i in range(config.n_splits):
        print(f'=== fold {i} of {config.n_splits} training ===')
        train_one_fold(data, i, data_args, config)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
