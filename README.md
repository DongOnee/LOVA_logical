# LOVA_logical
Hanyang Univ.

2019.3.6. dong's ver2.1
----
> batch size 를 placeholder 로 만들어서 실행시간을 좀더 짧게 만들었다.<br>
> embedding 하는 함수를 따로 만들어서 tensor graph 와 분리시킨 후 embedding 결과를 feeding 하는 식으로 하였다 <br>
> 거의 완성본이라고 생각해도 될거 같아.
- src/logical_train.py
  - training code 완성
  - ```python logical_train -e 100 -s 5```
  - ```-e``` : epoch
  - ```-s``` : checkpoint global step number

- src/util.py
  - 필요한 함수들 모아둠

- src/models.py
  - 필요한 모델 생성하는 코드들

- src/embedding.py
  - String 문단을 sentence 단위로 vectorize 하는 코드

- src/logical.py
  - 학습된 모델을 불러 들어와서 essay path 를 input 값으로 넣은후 점수 (0~1) 출력
  - 소요 시간 출력
  - ```python logical.py -e sample1.txt -s 2```
  - ```-e``` : essay file path
  - ```-s``` : checkpoint global step number


2019.1.21. dong's ver1.0
----
- src/logical_train.py
  - training code 완성
  - ```python logical_train -e 100 -s 5```
  - ```-e``` : epoch
  - ```-s``` : checkpoint global step number

- src/util.py
  - 필요한 함수들 모아둠

- src/models.py
  - 필요한 모델 생성하는 코드들

- src/logical.py
  - 학습된 모델을 불러 들어와서 essay path 를 input 값으로 넣은후 점수 (0~1) 출력
  - ```python logical.py -e sample1.txt -s 2```
  - ```-e``` : essay file path
  - ```-s``` : checkpoint global step number


2019.1.21. dong's ver1.0
----
내 노트북에선 너모 오래 걸려..

install package list
- tensorflow
- tensorflow-hub
- nltk
- numpy
- pandas

https://github.com/Rushikesh8983/MastersDataScience_Deep-learning-project/blob/master/RD_Language%20translate.ipynb
