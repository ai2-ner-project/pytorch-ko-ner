import sys
import argparse
import pickle

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

# kobert tokenizer/ model
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# Huggingface AutomModel/Tokenizer
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

# Custom Dataset
from ner.ner_dataset import NERDataset
from ner.ner_dataset import NERDatasetPreEncoded

from datasets import load_metric
from tqdm import tqdm

def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--test_file', required=True)
    p.add_argument('--use_AutoTokenizer', type=bool, default=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=16)

    config = p.parse_args()

    return config

# read data
def read_pickle(fn):
    with open(fn, 'rb') as f:
        dataset = pickle.load(f)
    data = pd.DataFrame(dataset.pop('data'))
    test_data = NERDatasetPreEncoded(data['input_ids'].tolist(), data['attention_mask'].tolist(), data['labels'].tolist())
    return test_data


# evaluation fuction
def compute_metrics(predictions, labels):
    metric = load_metric("seqeval")
    results = metric.compute(predictions=predictions, references=labels)

    return {"precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]}


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer_loader = AutoTokenizer if config.use_AutoTokenizer else KoBERTTokenizer
        tokenizer = tokenizer_loader.from_pretrained(train_config.pretrained_model_name)
        model = AutoModelForTokenClassification.from_pretrained(train_config.pretrained_model_name,
                                                                num_labels=len(index_to_label)
                                                               )
        model.load_state_dict(bert_best)

        if torch.cuda.is_available():
            model.cuda()
        device = next(model.parameters()).device

        test_data = read_pickle(config.test_file)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, return_tensors='pt')
        test_dataloader = DataLoader(test_data, collate_fn=data_collator, batch_size=config.batch_size, pin_memory=True)

        # Don't forget turn-on evaluation mode.
        model.eval()

        # Predictions
        predictions = []
        labels = []
        for batch in tqdm(test_dataloader):
            x = batch['input_ids']
            x = x.to(device)
            mask = batch['attention_mask']
            mask = mask.to(device)
            
            outputs = F.softmax(model(x, attention_mask=mask).logits, dim=-1)
            prediction = outputs.argmax(dim=-1)
            label = batch["labels"]
            
            predictions += prediction
            labels += label
            
        # Convert tensor to list and remove ignored index (special tokens)
        predictions = [prediction.tolist() for prediction in predictions]
        labels = [label.tolist() for label in labels]
        
        true_labels = [[index_to_label[l] for l in label if l != -100] for label in labels] 
        true_predictions = [[index_to_label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        
        print(compute_metrics(true_predictions, true_labels))

        for i in range(len(test_data)):
            sys.stdout.write('%s\t%s\n' % (tokenizer.convert_ids_to_tokens(test_data[i]['input_ids'], skip_special_tokens=True), true_predictions[i]))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
