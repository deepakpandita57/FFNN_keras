# Implement a Feed-Forward Neural Network for Part-of-Speech (POS) tagging using Keras

Task
=============================================================================================================
Implement a feed-forward POS tagger using keras. Network input should be a five word window centered on the current word.
A good default is that input words are represented with a 100 dimensional embedding, and the network has one hidden layer with 100 nodes.
You should experiment with different network structures.

Files
=============================================================================================================
"FFNN_keras.py" contains the code to train a feed-forward neural network for POS tagging
"README.md"

Instructions for running "FFNN_keras.py"
=============================================================================================================
To run the script "FFNN_keras.py" change the values of "train_file" and "test_file" variables in the script.
The script also takes the file containing all tags: "tag_file", "size_of_batch" and "no_of_epochs" as input.

Description
=============================================================================================================
The script reads train and test data and generates input for the network which is a five word window centered on the current word.
Vocab size considers all words from train and test data plus two special symbols for start(<S>) and stop(</S>).
Training and test instances as well as labels are converted into one-hot encoding.
The network has a embedding size of 100 and a hidden layer with 100 nodes and relu activation.
The output layer has 45(No. of tags) nodes and softmax activation.


Accuracy:
=============================================================================================================
Parameters:
Hidden layer nodes: 100
Output layer nodes: 45
size_of_batch: 5000
no_of_epochs: 10

I got an accuracy of 97.59% on training data and 94.52% on test data


Output:
Using TensorFlow backend.
Reading file: pos/train
Training instances: 950028
Labels: 950028
Reading file: pos/test
Test instances: 56684
Labels: 56684
Vocab size: 45498
(950028, 5)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 5, 100)            4549800
_________________________________________________________________
flatten_1 (Flatten)          (None, 500)               0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               50100
_________________________________________________________________
dense_2 (Dense)              (None, 45)                4545
=================================================================
Total params: 4,604,445
Trainable params: 4,604,445
Non-trainable params: 0
_________________________________________________________________
None
(950028, 1)
(950028, 45)

Epoch 1/10
950028/950028 [==============================] - 5s 5us/step - loss: 1.1926 - acc: 0.7374
Epoch 2/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.2055 - acc: 0.9378
Epoch 3/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.1528 - acc: 0.9513
Epoch 4/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.1305 - acc: 0.9579
Epoch 5/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.1156 - acc: 0.9625
Epoch 6/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.1042 - acc: 0.9662
Epoch 7/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.0950 - acc: 0.9693
Epoch 8/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.0873 - acc: 0.9718
Epoch 9/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.0808 - acc: 0.9740
Epoch 10/10
950028/950028 [==============================] - 4s 4us/step - loss: 0.0752 - acc: 0.9759
56684/56684 [==============================] - 6s 115us/step
[0.17668877461486859, 0.94520499612304032]

Some more experiments
========================================================================================================

1. 50 nodes in hidden layer - Training Acc: 97.46% Test Acc.: 94.78%
2. 2 Hidden layers with 100 nodes in each layer - Training Acc: 97.92% Test Acc.: 93.65%
3. 2 Hidden layers with 50 nodes in each layer - Training Acc: 97.96% Test Acc.: 94.29%
4. 2 Hidden layers with 100 nodes in the 1st layer and 50 in the 2nd - Training Acc: 97.86% Test Acc.: 94.38%
5. 3 Hidden layers with 100 nodes in each layer - Training Acc: 98.35% Test Acc.: 94.97%
6. 4 Hidden layers with 100 nodes in each layer - Training Acc: 98.05% Test Acc.: 94.13%
7. Embedding size: 200, 100 nodes in hidden layer - Training Acc: 98.07% Test Acc.: 94.48%
8. Embedding size: 300, 100 nodes in hidden layer - Training Acc: 98.21% Test Acc.: 94.39%


References
=============================================================================================================
This was done as a homework problem in the Statistical Speech and Language Processing class (CSC 448, Fall 2017) by Prof. Daniel Gildea (https://www.cs.rochester.edu/~gildea/) at the University of Rochester, New York.