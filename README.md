# PLM을 활용한 한국어 개체명 인식
주요 PLM의 fine-tuning을 통해 국립국어원 개체명 분석 말뭉치 2021에 대한 개체명 인식 다운스트림 (Downstream) 태스크를 진행했습니다. 데이터셋에서 개체명으로 정의된 15개 개체명 유형 (인명, 지명, 문명, 인공물 등)에 대해 대량 사전학습 언어모델 (PLM) 기반 개체명 인식기를 제공합니다. 

## Data
- 국립국어원  개체명  분석  말뭉치 2021 (https://corpus.korean.go.kr/main.do)
- 문어체 300만+ 문어체 300만 단어로 총 600만 단어, 약 80만 문장으로 구성
- 80만 문장 가운데 개체명 태깅 정보가 없는 경우를 제외하고 약 35만 문장으로 태스크 진행

## Pre-Requisite
- python 3.8
- 라이브러리 상세정보는 requirements.txt 파일 참고 
- skt/kobert-base-v1의 경우 kobert tokenizer 추가 설치 필요 

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf' 
```

## PLM Comparision
5가지 모델 비교

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
|skt/kobert-base-v1|0.771|0.954|
|klue/roberta-base|0.837|0.966|
|monologg/koelectra-base-v3-discriminator|0.830|0.965|
|monologg/kobigbird-bert-base|0.809|0.961|

## Reference
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- Kihyun Kim, Simple Neural Text Classification (NTC), GitHub
- 황석현 외, BERT를 활용한 한국어 개체명 인식기, 한국정보처리학회, 2019

## NER Demo
데모 url
