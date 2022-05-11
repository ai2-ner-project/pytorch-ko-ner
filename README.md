# ner
시스메틱 &amp; 패스트캠퍼스 기업연계 프로젝트 (NER 팀)

## Python
    (버전)
    python3.8
    
    (모듈)
    requirements.txt 내 참고
    kobert-tokenizer

## 가상환경 venv 활용 방법

    (Packages 업데이트)
    sudo apt update && sudo apt upgrade -y
    
    (Python3.8 설치)
    sudo apt install python3.8
    (Linux Ubuntu 기준) /usr/bin/python3.8
    
    (venv 설치)
    sudo apt install virtualenv
    
    (프로젝트로 이동)
    cd ner
    (venv 생성)
    virtual venv --python==/usr/bin/python3.8
    
    (venv 실행)
    . venv/bin/activate
    
    (python 확인)
    which python

## 파이썬 모듈 설치
    pip install -r requirements.txt
    pip install 'git+https://github.com/monologg/KoBERT-Transformers.git#egg=kobert_transformers'

## Reference
