#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:13:49 2022

@author: jelletuik
"""
# Here the libraries that are used, are loaded.
import csv
import os
import math
import glob
import nltk
import recengineCODE as rec
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('dutch'))

# Loop for csv files to turn them into a list.
def read_file_to_list():
    list_stories = []
    file = open("/Users/jelletuik/Downloads/development_material/stories.csv", newline='')
    content = csv.reader(file)
    for line in content:
        list_stories.append(line)
    file.close()
    return list_stories

def cosine(tfidf, query):
    products = []
    term_list = []
    query_list = []
    for row in tfidf[1:]:
        chosen_doc = row[tfidf]
        term_list.append(chosen_doc)
    for row in tfidf[1:]:
        chosen_query = row[query]
        query_list.append(chosen_query)
    for t1, q1 in zip(term_list, query_list):
        products.append(t1 * q1)
    return products
        
def cos_sim_below(doc, query):
    document_calc = rec.doc_length(doc)
    query_calc = rec.doc_length(query)
    calc_document = document_calc[1]
    doc_name = document_calc[0]
    calc_query = query_calc[1]
    calc_above = calc_document * calc_query
    acc_score = sum(cos_sim_top(doc, query))
    div_calc = acc_score / calc_above
    
    return [doc_name, div_calc]
    
    
    