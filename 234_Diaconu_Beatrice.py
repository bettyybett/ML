import csv
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import warnings
from scipy.io import wavfile as wav
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# caiile fisierelor
validation_path = 'C:\\Users\\Betty\\Desktop\\ml\\validation'
validation_labels = os.listdir(validation_path)
info_validation = []
train_audio_path = 'C:\\Users\\Betty\\Desktop\\ml\\train'
train_labels = os.listdir(train_audio_path)
train_info = []
test_audio_path = 'C:\\Users\\Betty\\Desktop\\ml\\test'
test_labels = os.listdir(test_audio_path)
test_info=[]

# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

# --------------TRAIN--------------

# for label in train_labels:
#     waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]

#functia de prelucrare a datelor
def wav_files(i):
    train_data, sample_rate = librosa.load(train_audio_path+'\\train\\'+i,sr = 22050)
    # fig = plt.figure(figsize=(14, 8))
    # librosa.display.waveplot(train_data, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=train_data, sr=sample_rate, n_mfcc=40)
    mfcc=np.mean(mfcc,axis=1)
    return mfcc
#fiecareui audiofile asociez datele aferente
with open("C:\\Users\\Betty\\Desktop\\ml\\train.txt","r") as file:
    for f in file:
        f = f.rstrip()
        audiofile,label=f.split(",")
        label=label.rstrip()
        audiodata= wav_files(audiofile) #aici apelez functia pentru audio ul meu
        train_info.append([audiodata,label]) #aici pastrez datele in lista
        #plt.show()


x_train = np.array(pd.DataFrame(train_info,columns=['data','label']).data.tolist()) #datele de intrare
y_train = np.array(pd.DataFrame(train_info,columns=['data','label']).label.tolist()) #datele de iesire
# --------------VALIDATION--------------
# for label in validation_labels:
#     waves_valid = [f for f in os.listdir(validation_path + '/' + label) if f.endswith('.wav')]

#functia de prelucrare a datelor
def wave_files(i):
    train_data, sample_rate = librosa.load(validation_path + '\\validation\\' + i,sr=22050)
    # fig = plt.figure(figsize=(14, 8))
    # librosa.display.waveplot(train_data, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=train_data, sr=sample_rate, n_mfcc=40)
    mfcc=np.mean(mfcc,axis=1)
    return mfcc

#fiecareui audiofile asociez datele aferente
with open("C:\\Users\\Betty\\Desktop\\ml\\validation.txt", "r") as file:
    for f in file:
        f = f.rstrip()
        audiofile, label = f.split(",")
        label = label.rstrip()
        audiodata = wave_files(audiofile) #asociez datele prin functia mea
        info_validation.append([audiodata, label]) #le pastrez in lista
        # plt.show()

x_validation = np.array(pd.DataFrame(info_validation,columns=['data','label']).data.tolist()) #datele de intrare
y_validation = np.array(pd.DataFrame(info_validation,columns=['data','label']).label.tolist()) #datele de iesire

#numarul minim de vecini posibili
neighbors=3
knn = KNeighborsClassifier(n_neighbors=neighbors)
#antrenarea datelor mele
knn.fit(x_train, y_train)
#prezicerea datelor si calcularea acuratetei
y_pred=knn.predict(x_validation)
print("accuracy:  ",metrics.accuracy_score(y_validation,y_pred))
# --------------TEST--------------
#functia de prelucrare a datelor
def wv_files(i):
    test_data, sample_rate = librosa.load(test_audio_path + '\\test\\' + i,sr=22050)
    # fig = plt.figure(figsize=(14, 8))
    # librosa.display.waveplot(train_data, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=test_data, sr=sample_rate, n_mfcc=40)
    mfcc=np.mean(mfcc,axis=1)
    return mfcc

#pentru ficare fisier audio asociez datele aferente
with open("C:\\Users\\Betty\\Desktop\\ml\\test.txt", "r") as file:
    for f in file:
        audiofile = f.rstrip()
        audiodata = wv_files(audiofile) #apelarea functiei aferenta
        test_info.append([audiodata,audiofile]) #le pastrez in lista

        # plt.show()


#salvarea datelor din test
x_test = np.array(pd.DataFrame(test_info, columns=['data', 'label']).data.tolist())
y_test = np.array(pd.DataFrame(test_info, columns=['data', 'label']).label.tolist())

#prezeicirea clasificari
y_pred=knn.predict(x_test)

#scierea etichetelor dupa predictie asociata fiecareui nume
with open('C://Users//Betty//Desktop//ml.csv', 'w', newline='') as csvfile:
    g = csv.writer(csvfile, delimiter=',')
    g.writerow(["name"] + ["label"])
    for i in range(len(y_pred)):
        g.writerow([test_info[i][1]] + [y_pred[i]])

# sigmoid_array = []
# for i in range(len(y_train)):
#     print(y_train[i])
#     sigmoid_array.append(sigmoid(float(y_train[i])))

#sigmoid_train(sigmoid_array,x_train,y_train)
#print(sigmoid_array)

# regressor = LinearRegression()
# regressor.fit(x_train,y_train)
# print(regressor.intercept_)
