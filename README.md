# PLM 기반 한국어 개체명 인식 (NER)
주요 한국어 PLM의 fine-tuning을 통해 한국어 개체명 인식 다운스트림 태스크를 진행했습니다. HuggingFace 라이브러리를 활용해 국립국어원 개체명 분석 말뭉치 데이터셋에서 개체명으로 정의된 15개 개체명 유형 (인명, 지명, 문명, 인공물 등)에 대해 개체명 인식기를 구현했습니다. 

## Data
- 국립국어원  개체명  분석  말뭉치 2021 (https://corpus.korean.go.kr/main.do)
- 문어체 300만 + 문어체 300만 단어로 총 600만 단어, 약 80만 문장으로 구성
- 80만 문장 가운데 개체명 태깅 정보가 없는 경우를 제외하고 약 35만 문장으로 태스크 진행

## Pre-Requisite
- python 3.8 기준으로 테스트
- 설치 모듈 상세정보는 requirements.txt 파일 참고 
- skt/kobert-base-v1의 경우 kobert tokenizer 추가 설치 필요 

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf' 
```

## PLM Comparision
|Model|PretrainingCorpus|Tokenization|Vocabulary|Hidden|Layers|Heads|Batch|
|-|-|-|-|-|-|-|-|
|klue/bert-base|6.5B words incl. Modu, Namuwiki|Mecab +BPE|32,000|768|12|12|256|
|klue/roberta-base|6.5B words incl. Modu, Namuwiki|Mecab +BPE|32,000|768|12|12|2048|
|skt/kobert-base-v1|Korean Wiki 54M words  |SentencePiece|8,002|3072|12|12|-|
|monologg/koelectra-base-v3-discriminator|crawled news data and Modu  |Wordpiece|35,000|768|12|12|256|
|monologg/kobigbird-bert-base|crawled news data and Modu  |Sentencepiece|32,500|768|12|12|32|

## How to Use

### Preparation
1. 여러 개의 json 파일로 저장된 데이터를 표 형식으로 변환하고 pickle로 저장합니다. pickle 파일 저장 시 이름은 load_path 경로에서 마지막 이름을 사용합니다.
```bash
python ./preprocess/json_to_tsv.py --load_path data/json/19_150tags_NamedEntity/NXNE2102008030.json --save_path data/raw
python ./preprocess/json_to_tsv.py --load_path data/json/21_150tags_NamedEntity --save_path data/raw
```
- '--load_fn' 다음에 특정 json 파일 또는 json 파일이 들어있는 폴더의 경로를 입력할 수 있습니다.
- '--return_tsv'를 추가하면 tsv 형식의 파일도 함께 저장합니다.
    
2. 학습과 평가 데이터셋을 만들어 train.pickle, test.pickle로 저장합니다.
데이터를 합치고 필요 없는 열, 문장이 누락된 행을 제거하는 과정이 포함됩니다.
```bash
python ./preprocess/preprocess.py --load_path data/raw --load_path data/dataset --test_size 0.15 --test_o_size 0.2 --save_all --return_tsv
```
```
4 files found :  ['SXNE21.pickle', 'SXNE2102007240.pickle', 'NXNE2102008030.pickle', 'NXNE2102203310.pickle']
file 0 :  351568
file 1 :  223962
file 2 :  150082
file 3 :  78170
|data before preprocessing| 803782
|data after preprocessing| 780546 / before dropping O sentences
|train| 663464 / |test| 117082 / before dropping O sentences
|train| 303028 / |test| 66845 / after dropping and sampling O sentences
```

- '--save_all'를 추가하면 분할하지 않은 전체 데이터도 파일로 추가로 저장합니다. 파일 이름은 data.pickle입니다.
- '--return_tsv'를 추가하면 tsv 형식의 파일도 함께 저장합니다.
- '--pass_drop_o'를 추가하면 NE를 포함하지 않는 문장도 데이터셋에 포함합니다. 기본적으로는 포함하지 않습니다.
- '--test_o_size'에서 정한 비율에 따라 평가 데이터에 NE를 포함하지 않는 문장을 추가합니다. 0.2를 입력하면 평가 데이터의 20%가 NE를 포함하지 않는 문장이 됩니다.
  NE를 포함하지 않는 문장을 뽑을 때 가능한 구어와 문어 문장의 비율을 반반으로 하되, 그렇게 하지 못할 경우 부족한 부분을 구어 문장으로 채웁니다.

3. 데이터셋을 선택한 모델의 토크나이저로 인코딩 합니다. 인코딩된 파일은 {원본 파일명}.{모델 이름}.encoded.pickle 이름으로 저장합니다.
```bash
python ./preprocess/encoding.py --load_fn data/dataset/train.pickle --save_path data/encoded
```
```
Tokenizer loaded : klue/roberta-base
Sentences encoded : |input_ids| 66845, |attention_mask| 66845
Token indices sequence length is longer than the specified maximum sequence length for this model (679 > 512). Running this sequence through the model will result in indexing errors
Sequence labeling completed : |labels| 66845
Encoded data saved as data/encoded/test.klue_roberta-base.encoded.pickle 
```

- '--with_text'를 추가하면 원문 문장을 포함하여 저장합니다.

### Train (Fine-Tuning)
인코딩이 완료된 데이터셋을 사용하여 학습을 진행합니다.
``` bash
python hf_trainer.py --model_fn models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle --pretrained_model_name klue/roberta-base --use_kfold --n_splits 5
```

- "--use_kfold"를 사용하는 경우 "--n_splits"와 사용할 폴드 수를 추가합니다.

### Fine-Tuning Details
- Total train data : 303028
- Train : validation = 8 : 2 (242422 : 60606)
- Batch size, epochs : 32, 2 (BigBird의 경우 16, 1)
- total iterations : 폴드별 15152번으로 동일하게 설정
- n-Fold : 5


### Inference
트레이닝과 동일하게 전처리한 테스트 데이터셋(66845개)에 대해 모델별, 폴더별 5개 체크포인트 결과의 평균을 최종값으로 하는 앙상블 기법을 적용합니다. --model_folder는 각 모델의 5개 체크포인트 결과가 들어있는 폴더이고, --test_file은 테스트 파일 이름입니다. 

```bash
python inference_ensemble.py --model_folder ./model -- test_file ./test_klue_roberta-base.encoded.pickle
```

## Evaluation
- Entity-level micro F1 (Entity F1) 
- 테스트 데이터의 인퍼런스 결과 klue/roberta-base가 근소한 성능으로 우수

|PLMs|F1 Score|Accuracy|
|-|-|-|
|klue/bert-base|0.831|0.965|
|klue/roberta-base|0.837|0.966|
|skt/kobert-base-v1|0.771|0.954|
|monologg/koelectra-base-v3-discriminator|0.830|0.965|
|monologg/kobigbird-bert-base|0.809|0.961|


## Reference
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- Kihyun Kim, Simple Neural Text Classification (NTC), GitHub
- 황석현 외, BERT를 활용한 한국어 개체명 인식기, 한국정보처리학회, 2019

## NER Demo
데모 url
