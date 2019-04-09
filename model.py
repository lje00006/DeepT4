import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 

from fasta_reader import readFile
from helpers import *
from plot import *

# Variable
maxlen = 150
batch_size = 40
epochs = 300
seq_rows, seq_cols = 20, maxlen
num_classes = 2



# build CNN model
model = Sequential()
model.add(Conv2D(50, kernel_size=(20,14),activation='relu'))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(650, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(lr=0.006),metrics=['accuracy'])

print('Loading training data...')
pos_Train = readFile("./data/pos_train.txt",maxlen)
neg_Train = readFile("./data/neg_train.txt",maxlen)

print('Generating labels and features...')
(y_train, x_train) = createTrainTestData(pos_Train,neg_Train,"Onehot")

print('Shuffling the data...')
index = np.arange(len(y_train))
np.random.shuffle(index)
x_train = x_train[index,:]
y_train = y_train[index]

x_train = x_train.reshape(x_train.shape[0],seq_rows, seq_cols,1)
y_train = keras.utils.to_categorical(y_train, num_classes)

print('Training...')
history = LossHistory()
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,callbacks=[history])

print('Plotting the acc-loss curve...')
history.loss_plot('epoch')

print('Loading test data...')
pos_Test = readFile("./data/pos_independent.txt",maxlen)
neg_Test = readFile("./data/neg_independent.txt",maxlen)

print('Generating labels and features...')
(y_test, x_test)=createTrainTestData(pos_Test,neg_Test,"Onehot")
x_test = x_test.reshape(x_test.shape[0],seq_rows,seq_cols,1)



print('Evaluating the model')
predicted_Probability = model.predict(x_test)
prediction = model.predict_classes(x_test)

print('Showing the confusion matrix')
cm=confusion_matrix(y_test,prediction)
print(cm)
print("ACC: %f "%accuracy_score(y_test,prediction))
print("F1: %f "%f1_score(y_test,prediction))
print("Recall: %f "%recall_score(y_test,prediction))
print("Pre: %f "%precision_score(y_test,prediction))
print("MCC: %f "%matthews_corrcoef(y_test,prediction))
print("AUC: %f "%roc_auc_score(y_test,prediction))
print('Plotting the ROC curve...')
plotROC(y_test,predicted_Probability[:,1])


print('saving the model...')
model.save_weights('./models_weights/model_weights.h5')
with open('./models_weights/json_model.json', 'w') as f:
     f.write(model.to_json())







