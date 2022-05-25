import os
import pickle
import argparse

import pandas as pd

# Import Tokenizer
from transformers import AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pretrained_model_name",
        required=True,
        help="Pre-trained model name to be going to train. Tokenizer will be assigned based on the model."
    )
    p.add_argument(
        "--load_fn",
        required=True,
        help="Original data to be going to preprocess."
    )
    p.add_argument(
        "--save_path",
        required=True,
        help="Original data to be going to preprocess."
    )
    p.add_argument(
        "--with_text",
        action="store_true",
        help="Return values with text for inference."
    )

    config = p.parse_args()

    return config


def get_char_labels(length, ne, label_to_index):
    char_labels = [0] * length
    for i in range(1, len(ne)+1):
        begin = ne[i]['begin']
        end = ne[i]['end']
        B_label = 'B-'+ne[i]['label'][:2]
        I_label = 'I-'+ne[i]['label'][:2]
        for j in range(begin, end):
            char_labels[j] = label_to_index[I_label]
        char_labels[begin] = label_to_index[B_label]

    return char_labels


def get_kobert_offset_mappings(kobert_tokens):
    kobert_offset_mappings = []
    offset_begin = -1
    for i in kobert_tokens:
        if i.startswith('▁'):
            offset_begin += 1
            i = i[1:]
    offset_end = offset_begin+len(i)
    kobert_offset_mappings.append((offset_begin, offset_end))
    offset_begin = offset_end

    return kobert_offset_mappings


def get_token_labels(char_labels, offset_mappings=None):
    token_labels = []

    for offset_mapping in offset_mappings:
        start_offset, end_offset = offset_mapping
        if start_offset == end_offset:
            token_labels.append(-100)
            continue
        token_labels.append(char_labels[start_offset])

    return token_labels


def BIO_tagging(text_tokens, ne, offset_map=None):
    labeled_sequence = [token if token in [
        '[CLS]', '[SEP]', '[PAD]'] else 'O' for token in text_tokens]
    ne_no = len(ne.keys())
    if ne_no > 0:
        for idx in range(1, ne_no+1):
            ne_dict = ne[idx]
            ne_dict_offset = ne_dict['begin']
            label_length = len(ne_dict['form'].replace(' ', ''))
            isbegin = True
            for word_idx, word in enumerate(text_tokens):
                if word == '[UNK]':
                    if offset_map[word_idx][0] == ne_dict_offset:
                        labeled_sequence[word_idx] = str(
                            'B-'+ne_dict['label'][:2])
                        break
                if label_length == 0:
                    break
                if ('##' in word) or ('▁' in word):
                    if word == '▁':
                        continue
                    word = word.replace('##', '')
                    word = word.replace('▁', '')
                if word in ne_dict['form']:
                    if isbegin:
                        labeled_sequence[word_idx] = str(
                            'B-'+ne_dict['label'][:2])
                        isbegin = False
                        label_length = label_length - len(word)
                        continue
                    elif (label_length > 0) & (isbegin == False) & (('B-' in labeled_sequence[word_idx-1]) or ('I-' in labeled_sequence[word_idx-1])):
                        labeled_sequence[word_idx] = str(
                            'I-'+ne_dict['label'][:2])
                        label_length = label_length - len(word)
                        continue

    return labeled_sequence


def get_label_dict(labels):
    BIO_labels = ['O']
    for label in labels:
        BIO_labels.append('B-'+label)
        BIO_labels.append('I-'+label)

    label_to_index = {label: index for index, label in enumerate(BIO_labels)}
    index_to_label = {index: label for index, label in enumerate(BIO_labels)}

    return label_to_index, index_to_label


def main(config):
    USE_KOBERT = True if config.pretrained_model_name == 'skt/kobert-base-v1' else False

    data = pd.read_pickle(config.load_fn)

    texts = data['sentence'].values.tolist()
    nes = data['ne'].values.tolist()

    pretrained_model_name = config.pretrained_model_name

    if USE_KOBERT:
        tokenizer_loader = KoBERTTokenizer
    else:
        tokenizer_loader = AutoTokenizer

    tokenizer = tokenizer_loader.from_pretrained(pretrained_model_name)
    print("Tokenizer loaded :", tokenizer.name_or_path)

    label_list = ["PS", "FD", "TR", "AF", "OG", "LC", "CV",
                  "DT", "TI", "QT", "EV", "AM", "PT", "MT", "TM"]
    label_to_index, index_to_label = get_label_dict(label_list)

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    max_length = 512

    if USE_KOBERT:
        encoded = tokenizer(texts,
                            add_special_tokens=True,
                            padding=False,
                            return_attention_mask=True,
                            truncation=True,
                            max_length=max_length)
    else:
        encoded = tokenizer(texts,
                            add_special_tokens=True,
                            padding=False,
                            return_attention_mask=True,
                            truncation=True,
                            max_length=max_length,
                            return_offsets_mapping=True)
    print("Sentences encoded : |input_ids| %d, |attention_mask| %d" %
          (len(encoded['input_ids']), len((encoded['attention_mask']))))

    ne_ids = []
    
    if USE_KOBERT:
        text_tokens = [[cls_token] + tokenizer.tokenize(text)[:max_length-2] + [sep_token] for text in texts]
        offset_mappings = [get_kobert_offset_mappings(text_token) for text_token in text_tokens]
    else: 
        offset_mappings = encoded['offset_mapping']

    for text, ne, offset_mapping in zip(texts, nes, offset_mappings):
        char_labels = get_char_labels(len(text), ne, label_to_index)
        token_labels = get_token_labels(char_labels, offset_mapping)
        ne_ids.append(token_labels)

    # if tokenizer.name_or_path == 'skt/kobert-base-v1':
    #     for text, ne in zip(texts, nes):
    #         text_tokens = [cls_token] + \
    #             tokenizer.tokenize(text)[:max_length-2] + [sep_token]
    #         ne_sequence = BIO_tagging(text_tokens=text_tokens, ne=ne)
    #         ne_id = [label_to_index[key] if key in label_to_index.keys()
    #                  else -100 for key in ne_sequence]
    #         ne_ids.append(ne_id)
    # else:
    #     for text, ne, offset_mapping in zip(texts, nes, encoded['offset_mapping']):
    #         text_tokens = [cls_token] + \
    #             tokenizer.tokenize(text)[:max_length-2] + [sep_token]
    #         ne_sequence = BIO_tagging(text_tokens, ne, offset_mapping)
    #         ne_id = [label_to_index[key] if key in label_to_index.keys()
    #                  else -100 for key in ne_sequence]
    #         ne_ids.append(ne_id)

    print("Sequence labeling completed : |labels| %d" % (len(ne_ids)))

    return_data = pd.DataFrame([encoded["input_ids"], encoded["attention_mask"], ne_ids, data['sentence_class'].values], index=[
                               "input_ids", "attention_mask", "labels", "sentence_class"]).T
    if config.with_text:
        return_data['sentence'] = texts

    label_info = {
        "label_list": label_list,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label
    }

    return_values = {
        "data": return_data.to_dict(),
        "label_info": label_info,
        "pad_token": (tokenizer.pad_token, tokenizer.pad_token_id),
        "pretrained_model_name": tokenizer.name_or_path
    }

    save_path = config.save_path   
    fn = os.path.split(config.load_fn)[1].split('.')[0]
    plm_name = pretrained_model_name.replace('/', '_')
    save_fn = os.path.join(save_path, f'{fn}.{plm_name}.encoded.pickle')

    with open(save_fn, "wb") as f:
        pickle.dump(return_values, f, 4)
    print("Encoded data saved as %s " % save_fn)

if __name__ == "__main__":
    config = define_argparser()
    main(config)
