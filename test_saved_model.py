from keras.models import model_from_json
from fasta_reader import readFile
from helpers import *
import numpy as np

maxlen = 150
seq_rows, seq_cols = 20, maxlen

print('Loading model...')

with open('./models_weights/json_model.json', 'r') as f:
     json_string = f.read()

model = model_from_json(json_string)
model.load_weights('./models_weights/model_weights.h5')

print('Loading data...')

testData = readFile("./data/S.cerevisiae-RBP354.txt",maxlen)

print('Generating features...')
x_test = createData(testData,"Onehot")
x_test = x_test.reshape(x_test.shape[0],seq_rows, seq_cols,1)


print('Predicting...')
predicted_Probability = model.predict(x_test)
prediction = model.predict_classes(x_test)

print('Saving the result..')

f = open ("result.txt", "w")
for i in prediction:
    if i == 1:
      f.write("T4SE\n")
    else:
      f.write("non-T4SE\n")

f.close()






