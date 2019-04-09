DeepT4
=========================

DeepT4 is a deep learning tool for Gram-negative bacteria Type IV secretion effectors prediction. The backbone of the framework is a convolutional neural network (CNN), which automatically extracts T4SEs-related features from 50 N-terminal and 100 C-terminal residues of the protein.

Author
=========================
ljs@swmu.edu.cn


Usage
=========================
To run: $ python model.py
The model file implements the convolutional neural network to train the model, evaluate the model, plot loss-acc and ROC cruves, and save the model. If you want to use different training and test data, please change the file name inside the file.

To run: $ python test_saved_model.py
The test_saved_model file predicts the T4SEs using saved CNN model. If you want to predict different data, please change the file name inside the file


Requirements
=========================
The latest (devel) version of DeepT4 requires£º 

python 3.5
Keras 2.2.4
numpy 1.16.2
matplotlib 3.0.2
scikit-learn  0.20.3



Reference: Li Xue, Bin Tang, Wei Chen,* and Jiesi Luo. A Deep Learning Framework for Sequence-Based bacteria type IV secreted effectors Prediction

 