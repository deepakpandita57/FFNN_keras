#!/usr/bin/python2

# Author: Deepak Pandita
# Date created: 28 Nov 2017


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import keras.preprocessing.text as text
import keras.utils as utils
import json
import numpy as np


train_file = '/data/train'
test_file = '/data/test'
#file containing all tags
tag_file = 'All_tags'
size_of_batch = 5000
no_of_epochs = 10

training_instances = []
training_labels = []
test_instances = []
test_labels = []
all_words = []
all_tags = []

#read tags
with open(tag_file) as tf:    
    all_tags = json.load(tf)

#Read train file using window of 5
print 'Reading file: '+train_file
f = open(train_file)
train_sentences = f.readlines()

for line in train_sentences:
	tokens = line.strip().split(' ')[1:]
	words = [y for x,y in enumerate(tokens) if x%2 == 0]
	tags = [y for x,y in enumerate(tokens) if x%2 != 0]
	for i, word in enumerate(words):
		instance = ''
		if i==0:
			instance += '<S> <S> '
			instance += words[i]
		if i==1:
			instance += '<S> '
			instance += words[i-1] + ' ' + words[i]
		if i > 1:
			instance += words[i-2] + ' ' + words[i-1] + ' ' + words[i]
		if i < (len(words)-2):
			instance += ' ' + words[i+1] + ' ' + words[i+2]
		if i==(len(words)-2):
			instance += ' ' + words[i+1]
			instance += ' </S>'
		if i==(len(words)-1):
			instance += ' </S> </S>'
		training_instances.append(instance)
		training_labels.append(tags[i])
		all_words.append(word)

print 'Training instances:',len(training_instances)
print 'Labels:',len(training_labels)

#Read test file using window of 5
print 'Reading file: '+test_file
f = open(test_file)
test_sentences = f.readlines()

for line in test_sentences:
	tokens = line.strip().split(' ')[1:]
	words = [y for x,y in enumerate(tokens) if x%2 == 0]
	tags = [y for x,y in enumerate(tokens) if x%2 != 0]
	for i, word in enumerate(words):
		instance = ''
		if i==0:
			instance += '<S> <S> '
			instance += words[i]
		if i==1:
			instance += '<S> '
			instance += words[i-1] + ' ' + words[i]
		if i > 1:
			instance += words[i-2] + ' ' + words[i-1] + ' ' + words[i]
		if i < (len(words)-2):
			instance += ' ' + words[i+1] + ' ' + words[i+2]
		if i==(len(words)-2):
			instance += ' ' + words[i+1]
			instance += ' </S>'
		if i==(len(words)-1):
			instance += ' </S> </S>'
		test_instances.append(instance)
		test_labels.append(tags[i])
		all_words.append(word)

print 'Test instances:',len(test_instances)
print 'Labels:',len(test_labels)

#Vocab size is considering all words from train and test data
vocab = set(all_words)
vocab_size = len(vocab) + 2
print 'Vocab size:',vocab_size

#convert to one-hot
encoded_train_instances = np.array([text.one_hot(inst,vocab_size, filters='\t\n') for inst in training_instances])
encoded_test_instances = np.array([text.one_hot(inst,vocab_size, filters='\t\n') for inst in test_instances])
print encoded_train_instances.shape

#Feed-Forward Neural Network
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=5))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(len(all_tags), activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print model.summary()

#convert labels to one hot
encoded_train_labels = np.array([text.one_hot(inst,len(all_tags), filters='\t\n') for inst in training_labels])
encoded_test_labels = np.array([text.one_hot(inst,len(all_tags), filters='\t\n') for inst in test_labels]) 
print encoded_train_labels.shape
one_hot_train_labels = utils.to_categorical(encoded_train_labels, num_classes = len(all_tags))
one_hot_test_labels = utils.to_categorical(encoded_test_labels, num_classes = len(all_tags))
print one_hot_train_labels.shape

#Training the model
model.fit(encoded_train_instances, one_hot_train_labels, epochs=no_of_epochs, batch_size=size_of_batch)
#Testing on test data
print model.evaluate(encoded_test_instances, one_hot_test_labels)
