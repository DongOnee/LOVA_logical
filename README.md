# LOVA_logical
Hanyang Univ.


2019.5.16. dong's ver 4.0
----
- src/embedding.py
    - 

2019.4.3. dong's ver3.0
----
> training 하는 과정에서 데이터를 전처리 하는데 오랜 시간이 걸리는걸 확인 하였다.<br>
> 이 부분을 해결 하기 위해 모든 데이터에 대해서 전처리를 해놓고 학습 시킬때 불러오는 식으로 하기로 하였다.

- src/train.py
    - 전처리 된 데이터를 불러들어와서 feed 시킨다.
    
- src/utils.py
    - 전처리 하는 함수를 새로 만들었다.
    
- src/preprocess.py
    - 기존의 코드에서 전처리 하는 코드.
    
- data/train_preproc_{}.csv
    - 전처리된 데이터

2019.3.29. dong's ver2.3
----

- src/embedding.py
    - 필요가 없다. 삭제
    
- src/utils.py
    - 기존에 embedding.py 에 쓰던것을 옮겨 놓았다. yield 를 이용한 방법이 메모리나 cpu 효율에 더 좋지 않을까 싶어서.

- src/train.py
    - 앞서 바꾼 내용에 맞게 수정하였다.

2019.3.18. dong's ver2.2
----

- src/logical_train.py -> src/train.py
    - 이름바꿈

- src/util.py && src/embedding.py
    - 기존에 있는 방법과 달리 바꿈

전체적으로 변경사항이 거의 없는 상태이다. 다양한 방법을 이용하여 모델을 저장한후 tensorflow js 에서 불러 오려고 노력했지만 실패 하였다. 관련 결과는 구글 공유 드라이브에 작성해 두었다.

2019.3.6. dong's ver2.1
----
> batch size 를 placeholder 로 만들어서 실행시간을 좀더 짧게 만들었다.<br>
> embedding 하는 함수를 따로 만들어서 tensor graph 와 분리시킨 후 embedding 결과를 feeding 하는 식으로 하였다 <br>
> 거의 완성본이라고 생각해도 될거 같다.
- src/logical_train.py
  - training code 완성
  - `python logical_train -e 100 -s 5`
  - `-e` : epoch
  - `-s` : checkpoint global step number

- src/util.py
  - 필요한 함수들 모아둠

- src/models.py
  - 필요한 모델 생성하는 코드들

- src/embedding.py
  - String 문단을 sentence 단위로 vectorize 하는 코드

- src/logical.py
  - 학습된 모델을 불러 들어와서 essay path 를 input 값으로 넣은후 점수 (0~1) 출력
  - 소요 시간 출력
  - `python logical.py -e sample1.txt -s 2`
  - `-e` : essay file path
  - `-s` : checkpoint global step number


2019.2.2. dong's ver1.2
----
- src/logical_train.py
  - training code 완성
  - `python logical_train -e 100 -s 5`
  - `-e` : epoch
  - `-s` : checkpoint global step number

- src/util.py
  - 필요한 함수들 모아둠

- src/models.py
  - 필요한 모델 생성하는 코드들

- src/logical.py
  - 학습된 모델을 불러 들어와서 essay path 를 input 값으로 넣은후 점수 (0~1) 출력
  - `python logical.py -e sample1.txt -s 2`
  - `-e` : essay file path
  - `-s` : checkpoint global step number


2019.1.21. dong's ver1.0
----
내 노트북에선 너모 오래 걸려..

install package list
- tensorflow : `conda install -c conda-forge tensorflow`
- tensorflow-hub : `conda install -c conda-forge tensorflow-hub`
- nltk : `conda install -c conda-forge nltk`
- numpy : `conda install -c conda-forge numpy`
- pandas : `conda install -c conda-forge pandas`
- pymongo : `conda install -c conda-forge pymongo`
- mongodb : `conda install -c conda-forge mongodb`

https://github.com/Rushikesh8983/MastersDataScience_Deep-learning-project/blob/master/RD_Language%20translate.ipynb

```
tensorflowjs_converter --help

tensorflowjs_converter \
--output_format=tfjs_graph_model \
 /Users/a01082705520/Documents/2018_Capstone/code/LOVA_logical/src/ \
 /Users/a01082705520/Documents/2018_Capstone/code/LOVA_logical/

tensorflowjs_converter \
--input_format=tf_saved_model \
--output_format=tfjs_graph_model \
/Users/a01082705520/Documents/2018_Capstone/code/LOVA_logical/src/convertModels \
/Users/a01082705520/Documents/2018_Capstone/code/LOVA_logical/src/

tensorflowjs_converter \
--input_format=tf_saved_model \
--output_format=tfjs_graph_model \
--saved_model_tags=serve \
builder_save .
```
