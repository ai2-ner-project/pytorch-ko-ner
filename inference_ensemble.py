import sys
import os
import argparse
import pickle

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

# kobert tokenizer/ model
from kobert_tokenizer import KoBERTTokenizer

# Huggingface AutomModel/Tokenizer
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

# Custom Dataset
from ner.ner_dataset import NERDataset
from ner.ner_dataset import NERDatasetPreEncoded

from tqdm import tqdm
from datasets import load_metric
from seqeval.metrics import classification_report


def define_argparser():
    """
    Define argument parser to take inference using pre-trained model.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--model_folder', required=True)
    p.add_argument('--test_file', required=True)
    p.add_argument('--use_AutoTokenizer', type=bool, default=True)
    p.add_argument('--gpu_id', type=int, default=-1) 
    p.add_argument('--batch_size', type=int, default=16)

    config = p.parse_args()
    return config

def read_pickle(fn):
    with open(fn, 'rb') as f:
        dataset = pickle.load(f)
    data = pd.DataFrame(dataset.pop('data'))
    test_data = NERDatasetPreEncoded(data['input_ids'].tolist(), data['attention_mask'].tolist(), data['labels'].tolist())
    return test_data

# inference function
def do_inference(tokenizer, model, bert_best, test_data):
    with torch.no_grad():
        tokenizer=tokenizer
        model = model
        model.load_state_dict(bert_best)

        if torch.cuda.is_available():
            model.cuda()
        device = next(model.parameters()).device
        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, return_tensors='pt')
        test_dataloader = DataLoader(test_data, collate_fn=data_collator, batch_size=config.batch_size, pin_memory=True)
                                                    
        # don't forget turn-on evaluation mode
        model.eval()

        # predictions
        logits = []
        labels = []
        for batch in test_dataloader: 
            x = batch['input_ids']
            x = x.to(device)
            mask = batch['attention_mask']
            mask = mask.to(device)
            logit = model(x, attention_mask=mask).logits
            logits.extend(logit.cpu().numpy())  # |len(test_data), length, classes|
            labels.extend(batch["labels"].cpu().numpy())  # |len(test_data), length|
    return logits, labels

# evaluation fuction
def compute_metrics(predictions, labels):
    metric = load_metric("seqeval")
    results = metric.compute(predictions=predictions, references=labels)

    return {"precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]}


def main(config):
    model_files = os.listdir(config.model_folder)
    saved_data_list = []
    for model_file in model_files:
        saved_data = torch.load(
            os.path.join(config.model_folder, model_file),
            map_location='cpu' if config.gpu_id < 0 else "cuda:%d" % config.gpu_id
        )
        saved_data_list.append(saved_data)

    train_config = saved_data_list[0]['config']
    index_to_label = saved_data_list[0]['classes']
    bert_best_list = [saved_data['bert'] for saved_data in saved_data_list]

    tokenizer_loader = AutoTokenizer if config.use_AutoTokenizer else KoBERTTokenizer
    tokenizer = tokenizer_loader.from_pretrained(train_config.pretrained_model_name)
    model = AutoModelForTokenClassification.from_pretrained(
                                                                train_config.pretrained_model_name,
                                                                num_labels=len(index_to_label))
    test_data = read_pickle(config.test_file)


    logits_list = []
    labels = None
    for bert_best in tqdm(bert_best_list):
        logits, tags = do_inference(tokenizer, model, bert_best, test_data)
        logits_list.append(logits)
        if labels == None:
            labels = tags
        
    final_logits = [(logits_list[0][i]+ logits_list[1][i]+ logits_list[2][i]+ logits_list[3][i] + logits_list[4][i])/5 for i in range(len(logits_list[0]))]

    predictions = [np.argmax(final_logit, axis=1) for final_logit in final_logits]
      
    # remove ignored index (special tokens)
    true_labels = [[index_to_label[l] for l in label if l != -100] for label in labels] 
    true_predictions = [[index_to_label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    print(compute_metrics(true_predictions, true_labels))
    print(classification_report(true_labels, true_predictions))

    # print predictions vs. labels
    for i in range(len(test_data)):
        sys.stdout.write('%s\t%s\n' % (tokenizer.convert_ids_to_tokens(test_data[i]['input_ids'], skip_special_tokens=True), true_predictions[i]))


if __name__ == "__main__":
    config = define_argparser()
    main(config)

