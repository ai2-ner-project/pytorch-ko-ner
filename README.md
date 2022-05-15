# PLM을 활용한 한국어 개체명 인식
주요 한국어 PLM의 fine-tuning을 통해 한국어 개체명 인식 다운스트림 태스크를 진행했습니다. HuggingFace 라이브러리를 활용해 국립국어원 개체명 분석 말뭉치 데이터셋에서 개체명으로 정의된 15개 개체명 유형 (인명, 지명, 문명, 인공물 등)에 대해 PLM 기반 개체명 인식기를 구현했습니다. 

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
|Model|TrainingCorpus|Tokenization|Vocabulary|Hidden|Layers|Heads|Batch|
|-|-|-|-|-|-|-|-|
|klue/bert-base|6.5B words incl. Modu, Namuwiki|Mecab +BPE|32,000|768|12|12|256|
|klue/roberta-base|6.5B words incl.Modu, Namuwiki  |Mecab +BPE|32,000|768|12|12|2048|
|skt/kobert-base-v1|Korean Wiki 54M words  |SentencePiece|8,002|3072|12|12|-|
|monologg/koelectra-base-v3-discriminator|crawled news data and Modu  |Wordpiece|35,000|768|12|12|256|
|monologg/kobigbird-bert-base|crawled news data and Modu  |Sentencepiece|32,500|768|12|12|32|

## How to Use

### Preparation
@simso - 어떤 전처리(json, tokenization, pickle 등)를 해서 어떤 형태의 데이터 입력

### Pretraining Details
@simso - bs, iter, ..

### Train
@simso - 기본 2 epochs * 5 folds로 진행, 폴드별 total iterations 15152번으로 맞춤 등등

### Inference
트레이닝과 동일하게 전처리한 테스트 데이터에 대해 모델의 폴더별 5개 체크포인트 결과의 평균을 최종값으로 하는 앙상블 기법을 적용했습니다. --model_folder는 각 모델의 5개 체크포인트 결과가 들어있는 폴더이고, --test_file은 테스트 파일 이름입니다. 

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
