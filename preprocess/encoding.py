import os
import pickle
import argparse

import pandas as pd

# Import Tokenizer
from transformers import AutoTokenizer
from transformers import ElectraTokenizer
from kobert_transformers.tokenization_kobert import KoBertTokenizer
from kobert_tokenizer import KoBERTTokenizer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pretrained_model_name",
        required=True,
        help="Pre-trained model name to be going to train. Tokenizer will be assigned based on the model."
    )
    p.add_argument(
        "--data_fn",
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


def BIO_tagging(text_tokens, ne, offset_map):
    labeled_sequence = [token if token in [
        '[CLS]', '[SEP]', '[PAD]', '[UNK]'] else 'O' for token in text_tokens]
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
                            ne_dict['label'][:2]) + '_B'
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
                            ne_dict['label'][:2]) + '_B'
                        isbegin = False
                        label_length = label_length - len(word)
                        continue
                    elif (label_length > 0) & (isbegin == False) & (('_B' in labeled_sequence[word_idx-1]) or ('_I' in labeled_sequence[word_idx-1])):
                        labeled_sequence[word_idx] = str(
                            ne_dict['label'][:2]) + '_I'
                        label_length = label_length - len(word)
                        continue

    return labeled_sequence


def get_label_dict(labels):
    BIO_labels = ['O']
    for label in labels:
        BIO_labels.append(label+'_B')
        BIO_labels.append(label+'_I')

    label_to_index = {label: index for index, label in enumerate(BIO_labels)}
    index_to_label = {index: label for index, label in enumerate(BIO_labels)}

    return label_to_index, index_to_label


def main(config):

    data = pd.read_pickle(config.data_fn)

    texts = data['sentence'].values.tolist()
    nes = data['ne'].values.tolist()

    pretrained_model_name = config.pretrained_model_name

    if pretrained_model_name == 'monologg/kobert':
        tokenizer_loader = KoBertTokenizer
    elif pretrained_model_name == 'skt/kobert-base-v1':
        tokenizer_loader = KoBERTTokenizer
    elif pretrained_model_name == 'monologg/koelectra-base-v3-discriminator':
        tokenizer_loader = ElectraTokenizer
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

    encoded = tokenizer(texts,
                        add_special_tokens=True,
                        padding=False,
                        return_attention_mask=True,
                        truncation=True,
                        max_length=max_length,
                        return_offset_mapping=True)
    print("Sentences encoded : |input_ids| %d, |attention_mask| %d" %
          (len(encoded['input_ids']), len((encoded['attention_mask']))))

    ne_ids = []
    # (batch_size, length)
    for text, ne, offset_mapping in zip(texts, nes, encoded['offset_mapping']):
        text_tokens = [cls_token] + \
            tokenizer.tokenize(text)[:max_length-2] + [sep_token]
        ne_sequence = BIO_tagging(text_tokens, ne, offset_mapping)
        ne_id = [label_to_index[key] if key in label_to_index.keys()
                 else -100 for key in ne_sequence]
        ne_ids.append(ne_id)

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
    }

    dir, fn = os.path.split(config.data_fn)
    fn = fn.split('.')[0]
    plm_name = '_'.join(pretrained_model_name.split('/'))
    encoded_fn = os.path.join(dir, f'{fn}.{plm_name}.encoded.pickle')

    with open(encoded_fn, "wb") as f:
        pickle.dump(return_values, f, pickle.HIGHEST_PROTOCOL)
    print("Encoded data saved as %s " % encoded_fn)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
