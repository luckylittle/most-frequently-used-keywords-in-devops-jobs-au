#!/bin/python3

import requests # used to grab data from the web
from bs4 import BeautifulSoup # used to parse HTML
from sklearn.feature_extraction.text import CountVectorizer # used to count number of words and phrases

texts = [] # initializing texts that will hold our job descriptions in this list

for index in range(0,1000,10): # go through 100 pages of Australian indeed
    page = 'http://au.indeed.com/jobs?q=solution+architect&start='+str(index) # identify the url of the job listings
    web_result = requests.get(page).text # use requests to actually visit the url
    soup = BeautifulSoup(web_result, 'html.parser') # parse the html of the resulting page
    for listing in soup.findAll('div', {'class':'summary'}): # for each listing on the page
        texts.append(listing.text) # append the text of the listing to our list

type(texts) # == list
vect = CountVectorizer(ngram_range=(1,2), stop_words='english') # get basic counts of one and two word phrases
matrix = vect.fit_transform(texts) # fit and learn to the vocabulary in the corpus
print(len(vect.get_feature_names())) # how many features are there

freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vect.vocabulary_.items()] #sort from largest to smallest
for phrase, times in sorted (freqs, key = lambda x: -x[1])[:25]:
    print(phrase, times) # print most used keywords and their frequency
