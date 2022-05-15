# NER - introduction
  프로젝트 소개

### Data
모두의 말뭉치 중 36만개

### Pretraining Details
bs, iter, ..

### Results
결과 표

## Pre-requisite

## Usage

### Preparation
  1. 여러 개의 json 파일로 저장된 데이터를 표 형식으로 변환하고 pickle로 저장합니다. pickle 파일 저장 시 이름은 load_path 경로에서 마지막 이름을 사용합니다.
      python ./preprocess/json_to_tsv.py --load_path data/json/19_150tags_NamedEntity/NXNE2102008030.json --save_path data/raw
      python ./preprocess/json_to_tsv.py --load_path data/json/21_150tags_NamedEntity --save_path data/raw
    - '--load_fn' 다음에 특정 json 파일 또는 json 파일이 들어있는 폴더의 경로를 입력할 수 있습니다.
    - '--return_tsv'를 추가하면 tsv 형식의 파일도 함께 저장합니다.
    
  2. 학습과 평가 데이터셋을 만들어 train.pickle, test.pickle로 저장합니다.
    데이터를 합치고 필요 없는 열, 문장이 누락된 행을 제거하는 과정이 포함됩니다.
      python ./preprocess/preprocess.py --load_path data/raw --load_path data/dataset --test_size 0.15 --test_o_size 0.2 --save_all --return_tsv

      4 files found :  ['SXNE21.pickle', 'SXNE2102007240.pickle', 'NXNE2102008030.pickle', 'NXNE2102203310.pickle']
      file 0 :  351568
      file 1 :  223962
      file 2 :  150082
      file 3 :  78170
      |data before preprocessing| 803782
      |data after preprocessing| 780546 / before dropping O sentences
      |train| 663464 / |test| 117082 / before dropping O sentences
      |train| 303028 / |test| 66845 / after dropping and sampling O sentences
    - '--save_all'를 추가하면 분할하지 않은 전체 데이터도 파일로 추가로 저장합니다. 파일 이름은 data.pickle입니다.
    - '--return_tsv'를 추가하면 tsv 형식의 파일도 함께 저장합니다.
    - '--pass_drop_o'를 추가하면 NE를 포함하지 않는 문장도 데이터셋에 포함합니다. 기본적으로는 포함하지 않습니다.
    - '--test_o_size'에서 정한 비율에 따라 평가 데이터에 NE를 포함하지 않는 문장을 추가합니다. 0.2를 입력하면 평가 데이터의 20%가 NE를 포함하지 않는 문장이 됩니다.
      NE를 포함하지 않는 문장을 뽑을 때 가능한 구어와 문어 문장의 비율을 반반으로 하되, 그렇게 하지 못할 경우 부족한 부분을 구어 문장으로 채웁니다.

  3. 데이터셋을 선택한 모델의 토크나이저로 인코딩 합니다. 인코딩된 파일은 {원본 파일명}.{모델 이름}.encoded.pickle 이름으로 저장합니다.
      python ./preprocess/encoding.py --load_fn data/dataset/train.pickle --save_path data/encoded

      Tokenizer loaded : klue/roberta-base
      Sentences encoded : |input_ids| 66845, |attention_mask| 66845
      Token indices sequence length is longer than the specified maximum sequence length for this model (679 > 512). Running this sequence through the model will result in indexing errors
      Sequence labeling completed : |labels| 66845
      Encoded data saved as data/encoded/test.klue_roberta-base.encoded.pickle 
    - '--with_text'를 추가하면 원문 문장을 포함하여 저장합니다.

### Train
  인코딩이 완료된 데이터셋을 사용하여 학습을 진행합니다.
      python hf_trainer.py --model_fn models --data_fn data/encoded/train.klue_roberta-base.encoded.pickle --pretrained_model_name klue/roberta-base --use_kfold --n_splits 5
  - "--use_kfold"를 사용하는 경우 "--n_splits"와 사용할 폴드 수를 추가합니다.

### Inference
  Fine-tuning 된 모델을 활용하여 개체명 인식을 수행합니다.
      python inference_ensemble.py --model_folder data/models/klue_roberta-base --test_file data/encoded/test.klue_roberta-base.encoded.pickle
  - 만약 skt/kobert-base-v1를 사용하는 경우에는 별도의 토크나이저를 불러와야 하므로 "--use_AutoTokenizer False"를 추가합니다.

## Reference

## NER Demo
