import torch
from torch.utils.data import Dataset


class NERCollator():

    def __init__(self, pad_token, with_text=True) -> None:
        self.pad_token = pad_token # ([PAD], {pad_token_id})
        self.with_text = with_text

    def __call__(self, samples):
        input_ids = [s['input_ids'] for s in samples]  # [CLS],[UNK],[SEP]가 포함되어있음
        attention_mask = [s['attention_mask'] for s in samples]
        labels = [s['labels'] for s in samples]

        # max_length 추출
        max_length = 0
        for line in input_ids:
            if max_length < len(line):
                max_length = len(line)

        # padding 추가
        for idx in range(len(input_ids)):
            # mini_batch내에 tokenize 된 문장(line)이 max_length보다 짧다면
            if len(input_ids[idx]) < max_length:
                # max_length = 원래 tokenize 된 문장(line) + ([PAD] x {max_length - len(원래 tokeniz 된 문장(line))})
                input_ids[idx] = input_ids[idx] + (
                    [self.pad_token[1]] * (max_length - len(input_ids[idx])))
                attention_mask[idx] = attention_mask[idx] + \
                    ([0] * (max_length - len(attention_mask[idx])))
                labels[idx] = labels[idx] + \
                    ([-100] * (max_length - len(labels[idx])))

        return_value = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

        return return_value


class NERDataset(Dataset):

    def __init__(self, texts, labels) -> None:
        self.texts = texts
        self.nes = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        ne = self.nes[item]

        return {
            'texts': text,
            'nes': ne,
        }


class NERDatasetPreEncoded(Dataset):

    def __init__(self, input_ids, attention_mask, labels) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        input_id = self.input_ids[item]
        attention_mask = self.attention_masks[item]
        label = self.labels[item]

        return {
            'input_ids': input_id,
            'attention_mask' : attention_mask,
            'labels': label,
        }