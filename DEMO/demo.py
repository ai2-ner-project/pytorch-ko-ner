
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from kobert_tokenizer import KoBERTTokenizer

FOLD = 5


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model_name_split = model_name.split('/')
    model = []
    for i in range(FOLD):
        if model_name=='monologg/kobigbird-bert-base':
            model.append(torch.load(f"/home/user/ner_project/ner/models/{model_name_split[0]}_{model_name_split[1].split('-')[0]}.1_epochs.512_length.{i}_fold.pth",
                                map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
        else:
            model.append(torch.load(f"/home/user/ner_project/ner/models/{model_name_split[0]}_{model_name_split[1].split('-')[0]}.2_epochs.512_length.{i}_fold.pth",
                                map_location='cuda:0' if torch.cuda.is_available() else 'cpu')) 
    return model

@st.cache(allow_output_mutation=True)
def load_token_and_model(select_plm):
    if select_plm=='skt/kobert-base-v1':
        temp_tokenizer = KoBERTTokenizer.from_pretrained(select_plm)
    else:
        temp_tokenizer = AutoTokenizer.from_pretrained(select_plm)
    temp_model = AutoModelForTokenClassification.from_pretrained(select_plm,num_labels=31)
    return temp_tokenizer, temp_model



def inference_for_demo(raw_text, plm_model, bert_best, tokenizer, index_to_label):
    with torch.no_grad():
        encoding = tokenizer(raw_text,return_tensors='pt')
        device = next(plm_model.parameters()).device
        tokenize_text = tokenizer.tokenize(raw_text)
        x = encoding['input_ids']
        x = x.to(device)
        mask = encoding['attention_mask']
        mask = mask.to(device)
        predictions = None

        for bert in bert_best:
            # evaluation mode,
            plm_model.eval()
            # Declare model and load pre-trained weights.
            plm_model.load_state_dict(bert,strict=False)

            # Take feed-forward
            y_hat=plm_model(x, attention_mask=mask).logits
            if predictions is None:
                predictions=y_hat
            else:
                predictions+=y_hat
            
    prediction = predictions /5.
    prediction = F.softmax(prediction, dim=-1)
    indice = torch.argmax(prediction,dim=-1)
    result = {}
    for i in range(1, len(indice[0])-1):
        result[tokenize_text[i-1]] = index_to_label[int(indice[0][i])]
    print(result)
    return result


def main():
    model_name = st.selectbox('Select PLM', ('klue/bert-base', 'klue/roberta-base', 'skt/kobert-base-v1',
                              'monologg/koelectra-base-v3-discriminator', 'monologg/kobigbird-bert-base'))

    saved_data = load_model(model_name)
    bert_best = [model['bert'] for model in saved_data]
    index_to_label = saved_data[0]['classes']

    tokenizer, plm_model = load_token_and_model(model_name)

    st.title("개체명 인식")

    activitied = ["NER Checker"]

    st.subheader("Input Text to Tokenize")
    raw_text = st.text_area("Enter Text Here", "Type Here")
    if st.button("Enter"):

        result = inference_for_demo(
                raw_text, plm_model, bert_best, tokenizer, index_to_label)
        st.write(result)


if __name__ == "__main__":
    main()
