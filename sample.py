# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from flask import Flask, jsonify, request,render_template
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

data = pickle.load( open( "PJ-assistant-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))
# Use pickle to load in the pre-trained model
#global graph
#graph = tf.get_default_graph()

with open(f'PJ-assistant-model.pkl', 'rb') as f:
    model = pickle.load(f)
    
def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # return tuple of intent and probability
    
    return return_list
import random
import json 
f = open('intents.json') 
datajson = json.load(f) 


# In[22]:
while True:
	inputtext=input("You:")
	for values in (datajson['intents']):
		if(classify_local(inputtext)[0]['intent']==values['tag']):
			print(random.choice(values['responses']))
			print(classify_local(inputtext))
			break


