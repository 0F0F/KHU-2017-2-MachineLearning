# KHU-2017-2-MachineLearning

rnn_train.py

터미널에서 parameter를 넣고 python3으로 실행합니다. -s: sequence_length -h: hidden_dimension -r: 사용할 dataset -c: 시행횟수
./ 에 model을 저장합니다.

external imports: 
os, tensorflow, numpy, matplotlib, sys, getopt, csv
********************************************************************

rnn_meta.py

rnn_train을 -h:(3,10), -s:(3, 16)의 범위에서 실행합니다.

external imports: 
os, datetime
********************************************************************

rnn_predict.py

train한 모델을 사용하기 위한 class predictor를 정의했습니다. 불러올 모델의 경로 model_path, 모델의 sequence_length와 hidden_dimension, 들어갈 데이터에 적절한 scale_base와 scale_factor를 넣어서 사용합니다.

external imports:
tensorflow, numpy, matplotlib, os
********************************************************************

rnn_verify.py

train한 모델이 유의미한 예측을 하는지 보기 위한 스크립트입니다. 거래소에서 약 10분간의 시세 데이터를 모은 real.csv파일에 대해 예측을 시도합니다.

external imports: 
numpy, math, matplotlib, sys
