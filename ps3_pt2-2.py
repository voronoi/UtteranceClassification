# -*- coding: utf-8 -*-
"""
Created on Mon Nov 03 09:22:03 2014

@author: aditya, jacha, tasso
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import metrics
#from operator import itemgetter
#from sklearn.metrics import classification_report
import os
import csv
#import pandas
#from gensim import corpora, models, similarities
#from itertools import chain
#import nltk
#from nltk.corpus import stopwords
#from operator import itemgetter
#import re

path = './data/' # Path to developer csv files
testPath = './testdata/' # Path to final test csv files
ratio = 0.80

files = os.listdir(path)
files = [f for f in files if f.find('.csv') > 0]

testFiles = os.listdir(testPath)
testFiles = [f for f in testFiles if f.find('.csv') > 0]

# List of tokens that can be removed for first task
markupTokens = {'{sl}', 'sp', '{ls}', '{lg}', '{cg}', '{ns}',
                '{br}', '*', '[', ']'}

# List of question words used to detect a question phrase
questionWords = {'why', 'what', 'how', 'when' ,'where', 'is', 'do',
                 'what\'s'}

full_set = []
for file in files:
    f = open(path + file,'rb')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        full_set.append(row)
        
# Held-out data set
finalTestSet = []
for file in testFiles:
    f = open(path + file,'rb')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        finalTestSet.append(row)
        
# Lists used for classification in task 2
X_train = []
y_train = []
X_test= []
y_test = []
finalTest = []
finalTestTruth = []
train_set = full_set[:int(len(full_set)*ratio)]
test_set = full_set[int(len(full_set)*ratio):]

# List to hold train_set list with tokens removed
cleanedTrain = []

# List to hold the Q/A for each line
trainResults = []
finalQAResults = []

# List to hold the E/M for each line
finalEMResults = []

# Used for next for loop
i = 0

# First task: decide on question/answer
for line in full_set:
    
    currentPhrase = line[5]
    
    # Clean up each line by removing tokens
    for token in markupTokens:
        if token in currentPhrase:
            currentPhrase = currentPhrase.replace(token, '')
            
    cleanedTrain.append(currentPhrase)
            
    firstThreeWords = currentPhrase.split()[:3]
#    print firstThreeWords
    
    # Assume the current line is an answer
    trainResults.append('A')
    
    # If a question word is found in the first three words of
    # the line, change the line to a question
    for q in questionWords:
        if q in firstThreeWords:
            trainResults[i] = 'Q'
            break
        
    if (len(currentPhrase.split()) > 20):
        trainResults[i] = 'A'
        
    i += 1

#numCorrect = 0

# Determine correctness in training set by comparing answer array
# with actual values from file
#for i in range(len(trainResults)):
#    if trainResults[i] == full_set[i][3]:
        #print train_set[i][5] + trainResults[i]
#        numCorrect += 1
#    else:
#        print full_set[i]

#print str(numCorrect) + " Question/Answer Correctly Classified"
#print str((float(numCorrect)/len(full_set)) * 100) + "% accuracy"

i = 0

for line in finalTestSet:
    
    currentPhrase = line[5]
    
    # Clean up each line by removing tokens
    for token in markupTokens:
        if token in currentPhrase:
            currentPhrase = currentPhrase.replace(token, '')
            
    firstThreeWords = currentPhrase.split()[:3]
#    print firstThreeWords
    
    # Assume the current line is an answer
    finalQAResults.append('A')
    
    # If a question word is found in the first three words of
    # the line, change the line to a question
    for q in questionWords:
        if q in firstThreeWords:
            finalQAResults[i] = 'Q'
            break
        
    if (len(currentPhrase.split()) > 20):
        finalQAResults[i] = 'A'
        
    i += 1
    
numCorrect = 0

for i in range(len(finalQAResults)):
    if finalQAResults[i] == finalTestSet[i][3]:
        #print train_set[i][5] + trainResults[i]
        numCorrect += 1
    else:
        print finalTestSet[i]

print str(numCorrect) + " Question/Answer Correctly Classified"
print str((float(numCorrect)/len(finalTestSet)) * 100) + "% accuracy"

############# Beginning of E/M classification #################
for line in train_set:
    
    currentTrainFeature = line[5]
    #currentTrainFeature = ' '.join((line[0], line[1], line[2], line[5]))
    #currentTrainFeature = ' '.join((line[0], line[2], line[5]))
    currentTrainGT = line[4]
    X_train.append(currentTrainFeature)
    y_train.append(currentTrainGT)
    
for line in test_set:
    
    #currentTestFeature = line[5]
    currentTestFeature = ' '.join((line[0], line[1], line[2], line[5]))
    currentTestGT = line[4]
    X_test.append(currentTestFeature)
    y_test.append(currentTestGT)
    
for line in finalTestSet:
    
    currentFTestFeature = line[5]
    #currentFTestFeature = ' '.join((line[0], line[1], line[2], line[5]))
    #currentFTestFeature = ' '.join((line[0], line[2], line[5]))
    currentFTestGT = line[4]
    finalTest.append(currentFTestFeature)
    finalTestTruth.append(currentFTestGT)

#Convert Ground Truth To Array for use with the SKLEARN Metrics MODULE

import numpy as np
testGTarray = np.asarray(y_test)

# Multinomial Naive Bayes
vectorizer = TfidfVectorizer(min_df=2, 
 ngram_range=(1, 2), 
 stop_words='english', 
 strip_accents='unicode', 
 norm='l2')
 
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
finalTest = vectorizer.transform(finalTest)
nb_classifier = MultinomialNB().fit(X_train, y_train)

'''
#Multinomial Naive Bayes
y_nb_predicted = nb_classifier.predict(X_test)
finalNBPredicted = nb_classifier.predict(finalTest)
'''

'''
# SVM
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
y_svm_predicted = svm_classifier.predict(X_test)   
finalSVMPredicted = svm_classifier.predict(finalTest)
'''
from sklearn.svm import SVC
linear_svc = SVC(kernel='linear')
finalSVCPredicted = linear_svc.fit(X_train, y_train).predict(finalTest) 

'''
# Logistic Regression
from sklearn.linear_model import LogisticRegression
maxent_classifier = LogisticRegression().fit(X_train, y_train)
y_maxent_predicted = maxent_classifier.predict(X_test)
finalMaxEntPredicted = maxent_classifier.predict(finalTest)
'''

#y_train = np.array([el for el in nyt_labels[0:trainset_size]])
#
#X_test = np.array([''.join(el) for el in nyt_data[trainset_size+1:len(nyt_data)]]) 
#y_test = np.array([el for el in nyt_labels[trainset_size+1:len(nyt_labels)]]) 

# Get the majority prediction from the 3 classifiers above for each line
# and use it as the classification

j = 0

for truth in finalTestTruth:
    numM = 0
    numE = 0
    '''
    currentPredictions = [finalNBPredicted[j], finalSVMPredicted[j],
                   finalMaxEntPredicted[j]]
    for prediction in currentPredictions:
        if prediction == 'M':
            numM += 1
        else: numE += 1
    if numM > 1:
        bestPrediction = 'M'
    else: bestPrediction = 'E'
    '''
    bestPrediction = finalSVCPredicted[j]
    finalEMResults.append(bestPrediction)
    
    #print currentPredictions, bestPrediction
    j += 1
    
emcor = 0
for i, r in enumerate(finalEMResults):
    if finalEMResults[i] == finalTestSet[i][4]:
        emcor += 1
    #else: print str(i) + ": " + finalEMResults[i] + " " + finalTestSet[i][4]
    
#print "Test set: " + str(len(finalTestSet))
print str(emcor) + " Emotional/Material Correctly Classified"
print str((float(emcor)/len(finalTestSet)) * 100) + "% accuracy"
    
# Write results out to a .csv file in the same format as input
if not os.path.exists('./result/'):
    os.makedirs('./result/')
resultFile = open('./result/results.csv', 'w+')

j = 0

for line in finalTestSet:
    resultFile.write(line[0] + ',' + line[1] + ',' + line[2] + ',' +
            finalQAResults[j] + ',' + finalEMResults[j] +
            ',' + line[5] + '\n')
    j += 1

resultFile.close()
#print finalEMResults
#print finalQAResults