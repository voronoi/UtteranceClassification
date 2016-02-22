# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:29:53 2014
@author: jascha, tasso
"""

import os
import csv
import nltk, sklearn
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

path = './data/' # Path to csv files
ratio = 0.70

files = os.listdir(path)
files = [f for f in files if f.find('.csv') > 0]

# lump all the files together
full_set = []
for file in files:
    f = open(path + file,'rb')
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        full_set.append(row)
    f.close()

# Split up the lumped set into dev train and test
train_set = full_set[:int(len(full_set)*ratio)]
test_set = full_set[int(len(full_set)*ratio):]

###
### First task: decide on question/answer
###

# List of tokens that can be removed for first task
markupTokens = {'{sl}', 'sp', '{ls}', '{lg}', '{cg}', '{ns}',
                '{br}', '*', '[', ']'}

# List of question words used to detect a question phrase
questionWords = {'why', 'what', 'how', 'when' ,'where', 'is', 'do',
                 'what\'s'}

# List to hold train_set list with tokens removed
cleanedTrain = []

# List to hold the Q/A for each line
qaResults = []

# Apply the simple rule-based approach (on the entire set)
for i, line in enumerate(full_set):
    
    currentPhrase = line[5]
    
    # Clean up each line by removing tokens
    for token in markupTokens:
        currentPhrase = currentPhrase.replace(token, '')
            
    cleanedTrain.append(currentPhrase)
            
    firstThreeWords = currentPhrase.split()[:3]
    
    # Assume the current line is an answer
    qaResults.append('A')
    
    # If a question word is found in the first two words of
    # the line, change the line to a question
    for q in questionWords:
        if q in firstThreeWords:
            qaResults[i] = 'Q'
    
    # However, if the line is particulary long, we know it is an answer.
    if (len(currentPhrase.split()) > 20):
        qaResults[i] = 'A'

###
### Second task: Classify emotional/material
###

# Ground truth for E/M and text, filtered text
train_em = [l[4] for l in train_set]
train_text = [l[5] for l in train_set]

# clean training text
for token in markupTokens:
        train_text = [l.replace(token, '') for l in train_text]

# Split up text into a list for other operations
train_text_list = [nltk.wordpunct_tokenize(l) for l in train_text]
for i, line in enumerate(train_text_list):
    train_text_list[i] = [w for w in train_text_list[i] if w.isalpha()]
    train_text_list[i] = [w for w in train_text_list[i] if w not in stopwords.words('english')]

test_em = [l[4] for l in test_set]
test_text = [l[5] for l in test_set]
#train_text_filtered = [l for l in train_text if ]

# Tf Idf is probably inappropriate in this case
vectorizer = TfidfVectorizer(min_df=2, 
                             ngram_range=(1, 2), 
                             stop_words='english', 
                             strip_accents='unicode', 
                             norm='l2')

train_tfidfVectors = vectorizer.fit_transform(train_text)
test_tfidfVectors = vectorizer.transform(test_text)

nb_classifier = MultinomialNB().fit(train_tfidfVectors, train_em)
emResults = nb_classifier.predict(test_tfidfVectors)


# Wordnet hyponym features
emotion = wn.synsets('emotion')

#for i, line in train_text_list:
#    break


#
# Summarize results
#
qaCorrect, emCorrect = 0, 0

for i, r in enumerate(qaResults):
    if qaResults[i] == full_set[i][3]:
        #print train_set[i][5] + qaResults[i]
        qaCorrect += 1
    else: # but why would it ever be wrong?
        print full_set[i]

for i, r in enumerate(emResults):
    if emResults[i] == test_set[i][4]:
        #print train_set[i][5] + qaResults[i]
        emCorrect += 1
#    else: # but why would it ever be wrong?
#        print full_set[i]

print str(qaCorrect) + " Question/Answer Correctly Classified"
print str((float(qaCorrect)/len(full_set)) * 100) + "% accuracy"
print str(emCorrect) + " Emotional/Material Correctly Classified"
print str((float(emCorrect)/len(test_set)) * 100) + "% accuracy"

# Write to file

#results_set = full_set
#for i, l in results_set:
    

#f = open(path + 'output.csv','rb')
#writer = csv.writer(f, delimiter=',')
#for l in results_set:
#    writer.writerow(l)
#f.close()
