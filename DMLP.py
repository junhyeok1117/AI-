import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn import datasets
from sklearn.datasets import load_diabetes

diabetse=load_diabetes()
print(diabetse.keys())
print(diabetse.feature_names)
aa=diabetse.data 
bb=diabetse.target

nr,nc=aa.shape 

from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential

nov=400 
x_train=aa[:nov,:] # 입력데이터
y_train=bb[:nov] # 라벨 값

model=Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam') #모델 컴파일
model.summary()

H=model.fit(x_train,y_train,batch_size=20,epochs=500) 
plt.figure()
plt.title('loss')
plt.plot(H.history['loss'])
