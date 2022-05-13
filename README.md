# PLM을 활용한 한국어 개체명 인식 (Named Entity Recognition)

주요 PLM들의 fine-tuning을 통해 국립국어원 개체명 분석 말뭉치 2021에 대한 개체명 인식 다운스트림 태스크를 진행했습니다. 데이터셋에서 개체명으로 정의된 15개 개체명 유형 (인명, 지명, 문명, 인공물 등)에 대해 대량 사전학습모델 (PLM) 기반 개체명 인식기를 제공합니다. 

## Data

- 국립국어원  개체명  분석  말뭉치 2021 (https://corpus.korean.go.kr/main.do)
- 문어체 300만+ 문어체 300만 단어로 총 600만 단어, 약 80만 문장으로 구성
- 80만 문장 가운데 개체명 태깅 정보가 없는 경우를 제외하고 약 35만 문장으로 데이터셋 구성

## Pre-Requisite

- python 3.8
- 설치 모듈 상세정보는 requirement.txt 파일 참고 
- skt/kobert-base-v1의 경우 kobert tokenizer 추가 설치 필요 

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf' 
```

## PLM Comparision
5가지 모델 비교

## Usage

### Preparation
@simso - 어떻게 전처리(json, tokenization, pickle 등)해서 어떤 형태의 데이터를 집어 넣었는지

### Pretraining Details
@simso - bs, iter, ..

### Train
@simso 

### Inference

## Evaluation
평가지표 엔티티 F1, 테스트 데이터셋 결과

## Reference
참고논문 및 깃허브 

## NER Demo
데모 url
