#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 14:08:24 2022

@author: jelletuik
"""


from flask import Flask, flash, escape, request, render_template, redirect, url_for, send_from_directory

from werkzeug.utils import secure_filename

#from nltk.corpus import stopwords

import re

import os

import math

import glob

import recengineCODE as rec

import shutil


app = Flask(__name__)

# When the function below is called, Dutch stopwords will be read
# from a text file, seperated with backspace. It returns a list with stops
def read_stopwords():
    with open('stopwords.txt') as stops:
        content = stops.read()
        handle_backspace = content.split('\n')
        
    return handle_backspace

# This function is used within 
def square(number):
    return (number * number)

# This function is used for the function 'dot_product'. It returns via list
# comprehension, the sum of the product between document and query
def column_sum(document):
    return [sum(i) for i in zip(*document)]


# Process_text is used to read and process one document at a time.
# It returns a nested dictionary with the document name (main key) and word
# frequencies (subkey) and words (subvalue). 
def process_text(document):
    punctuation= '''!()-[]{};:'"\, <>./?@+#$%^&*_~'''
    stops = set(read_stopwords())


    try:
        doc_open = open(document, 'r')
        doc_name = os.path.basename(document)
    except:
        doc_open = document
        doc_name = 'query'
    
    non_capital_list = []
    cleaned_list = []
    revised_list = []
    end_list = []
    dev_list = []
    words_count = {}
   
    for lines in doc_open:
        for words in lines.split():
            remove_capitals = words.lower()
            non_capital_list.append(remove_capitals)
    for punc in non_capital_list:
        for character in punc.split():
            remove_punctuation = character.strip(punctuation)
            cleaned_list.append(remove_punctuation)
    for word in cleaned_list:
        if word not in stops:
            revised_list.append(word)
    for word in revised_list:
        for letter in word.split():
            short_word = letter.translate({ord(i): None for i in punctuation})
            end_list.append(short_word)
    for word in end_list:
        for letter in word.split():
            if len(letter) >= 3: 
                root_word = letter.rstrip("s")
                dev_list.append(root_word)
            else:
                dev_list.append(letter)
    for word in dev_list:
        for amount in word.split():
            words_count[word] = words_count.get(word, 0)+1
    
    my_dict = {doc_name:words_count}
    return my_dict

# When this function is called, it will read all document from the current 
# working directory. In this way, it can read an arbitrary set of documents.
# When returned, the variable 'files_dict' contains a nested dictionary, with
# the name of documents as keys and words / frequency as values.
def read_all_docs():
    files_dict = {}
    cwd = os.getcwd()
    path = cwd + '/collection/*.txt'
    for files in glob.glob(path):
        with open(files) as texts:
            temp_dict = process_text(files)
        files_dict.update(temp_dict)
    return files_dict

# Process_upload is a shortened version of process_text, and is used for
# the document previews on the result page. It returns a list with words from
# the documents in right order. 
def process_upload(document):
    punctuation= '''!()-[]{};:'"\, <>./?@+#$%^&*_~'''
    stops = set(read_stopwords())


    try:
        doc_open = open(document, 'r')
        doc_name = 'query'
    except:
        doc_open = document
        doc_name = 'query'
    
    non_capital_list = []
    cleaned_list = []
    revised_list = []

   
    for lines in doc_open:
        for words in lines.split():
            remove_capitals = words.lower()
            non_capital_list.append(remove_capitals)
    for punc in non_capital_list:
        for character in punc.split():
            remove_punctuation = character.strip(punctuation)
            cleaned_list.append(remove_punctuation)
    for word in cleaned_list:
        if word not in stops:
            revised_list.append(word)
            
    return revised_list

# Unique_terms will return a set of all exclusive terms from the documents.
def unique_terms(nested_dict):
    set_dict = set()
    for doc in nested_dict:
        for word in nested_dict[doc]:
            set_dict.add(word)
    return set_dict
    
# This function creates an empty matrix with the name 
# of documents and columns
def term_freq_matrix(nested_dict, bow):
    columns = ['term']
    matrix = []
    unique_list_terms = list(bow)
    for doc in nested_dict:
        columns.append(doc)
    matrix.append(columns)
    for term in range(len(unique_list_terms)):
        all_terms = unique_list_terms[term]
        matrix.append([all_terms])
    return matrix

# This function will fill the matrix with term frequencies
def add_matrix(matrix_empty, nested_dict, bow):
    document_list = matrix_empty[0]
    unique_words = list(bow)
    document_matrix =[]
    document_matrix.append(document_list)
    
    for line in range(len(unique_words)):
        half_list = [unique_words[line]]
        for doc in range(len(document_list)):
            if document_list[doc] in nested_dict:
                sub_document = nested_dict.get(document_list[doc])
                if unique_words[line] in sub_document:
                    sub_words = sub_document.get(unique_words[line])
                    half_list.append(sub_words)
                else:
                    half_list.append(0)
        document_matrix.append(half_list)
    return document_matrix

# idf is the last function in preprocessing. When called, it will
# calculate the inverse document frequencies and create the tfidf matrix
def idf(base_matrix):
    matrix_between = base_matrix
    amended_matrix = matrix_between
    N = len(base_matrix[0]) - 1
    idf_list = [0]
    for row in range(1, len(base_matrix)):
        revised = matrix_between[row]
        df = 1
        df_list = []
        for column in range(1, len(base_matrix[row])):
            if revised[column] > 0:
                df += 1
                df_list.append(1)
            else:
                df += 0
                df_list.append(0)
        df_total = sum(df_list)
        idf_list.append(df_total)
    for idf_value in range(1, len(matrix_between)):
        calculate = idf_list[idf_value]
        formula_idf = math.log2(N/calculate)
        line = matrix_between[idf_value]
        for term in range(1, len(base_matrix[idf_value])):
            value = line[term] * formula_idf
            amended_matrix[idf_value][term] = value
    return amended_matrix

# The function below will calculate the dot product for all documents against
# the query. It returns a table with the sum for each document
def dot_product(collection, query=None):
    query_vector = []
    document_vector = []
    col_length = len(collection) - 1
    product_matrix = []
    for q in range(len(collection)):
        q_value = collection[q][-1]
        query_vector.append(q_value)
    
    for document in range(1, len(collection)):
        tempdoc = collection[document]
        product_document = []
        for value in range(1, len(collection[0])):
            word_value = collection[document][value]
            query_value = query_vector[document]
            product = word_value * query_value
            product_document.append(product)
        product_matrix.append(product_document)

    product_total = column_sum(product_matrix)
    product_total_table = []
    product_total_table.append(collection[0][1:])

        
    return product_total
            
# Cosine uses the document length function from the main program.
# It returns a table with the similarities for each document.
def cosine(collection):
    run_document_length = rec.document_length(collection)
    run_dot_product = dot_product(collection)
    length_x = []
    for document in range(1, len(run_document_length)):
        q_value = run_document_length[-1][0]
        d_value = run_document_length[document][0]
        tempresult = q_value * d_value
        length_x.append(tempresult)
    

    similarity = []

    for score in range(len(run_dot_product)):
        var_score = run_dot_product[score]
        var_result = var_score/length_x[score]
        similarity.append(var_result)

        
    return similarity


def show_output(collection):
    results = []
    tempmtx = cosine(collection)
    

    for score in range(len(tempmtx)):
        sublist = []
        document_name = collection[0][score+1]
        document_similarity = tempmtx[score]
        sublist.append(document_name)
        sublist.append(document_similarity)
        results.append(sublist)

    for cos in range(len(results)):
        if results[cos][1] < 0.25:
            results[cos].insert(2,'low')
        if results[cos][1] >= 0.25:
            if results[cos][1] < 0.5:
                results[cos].insert(2,'medium')
        if results[cos][1] >= 0.5:
            results[cos].insert(2,'high')

    ranked_list = sorted(results, key = lambda x: x[1], reverse = True)
    return ranked_list



def convert_string(sentence):
    li_in = list(sentence.split(' '))
    return li_in

def search_sentence(document):
    file = open(document, 'r')
    strings = re.findall(r"\w+|\W+", file.read(), flags =re.DOTALL)
    return strings


def document_preview(collection, query):
    cwd = os.getcwd()
    path_doc = cwd + '/collection/'
    whole_path = glob.glob(path_doc + '*.txt')
    sentence_result = []
    ranked_mtx = show_output(collection)
    preview_list = ranked_mtx[1:6]
    newlist = preview_list
    q_set = set(query)
    qu_li = list(q_set)

    
    relevant_documents = []

    
    for hits in preview_list:
        relevant_documents.append(hits[0])

        
    directory = []
    for path in relevant_documents:
        direction = os.path.abspath(path)
        directory.append(direction)
        

    preview = []    
    for doc in range(len(relevant_documents)):
        sentence = search_sentence(relevant_documents[doc])
        newlist[doc].insert(0, doc+1)
        for q in qu_li:
            if q in sentence:
                keyword = sentence.index(q)
                short = sentence[keyword-10:keyword+10]
                string_sentence = "".join(short)
                newlist[doc].append(sentence[keyword])
                adjusted = string_sentence.replace('\n',' ')
                newlist[doc].append(adjusted)
                
    return newlist


cwd = os.getcwd()

def prepare_query(collection, query):
    prepared_query = collection
    prepared_query[0].append('query')
    for term in range(1, len(collection)):
        prepared_query[term].append(0)
        for k, v in query.items():
            for key in v:
                if key == collection[term][0]:
                    prepared_query[term][-1] = v.get(key)
                else:
                    continue
                    
    return prepared_query

def coll_x_coll(document):
    tfidf = rec.load_matrix()
    tfidf_col = tfidf
    position = tfidf[0].index(document)
    tfidf_col[0].append(document)
    for doc in range(1,len(tfidf)):
        tfidf_col[doc].append(tfidf_col[doc][position])
        
    return tfidf_col

UPLOAD_FOLDER = cwd + '/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


@app.route('/', methods=['GET', 'POST'])
def homepage():
    path = cwd + '/collection'
    dirs = os.listdir(path)
    temp = []
    for dir in dirs:
        temp.append({'name' : dir})
    return render_template('homepage.html', documents=temp)

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    collection_doc = read_all_docs()
    collectiondoc_query = collection_doc
    uploaded_file = request.files['file']
    file_name = uploaded_file.filename

 
    if request.method == 'POST':
        uploaded_file.save(os.path.join(UPLOAD_FOLDER,secure_filename(uploaded_file.filename)))
        path_doc = cwd + '/uploads/'
        whole_path = glob.glob(path_doc + '*.txt')
        whole_path_string = ''.join(whole_path)
        user_doc_processing = process_text(whole_path_string)
        pre_preview = process_upload(whole_path_string)
#        test_collection = read_all_docs() 
#        collection_without_query = test_collection
#        collection_without_query.update(user_doc_processing)
#        terms_set = unique_terms(collection_without_query)
#        empty_matrix = term_freq_matrix(collection_without_query, terms_set)
#        filled_matrix = add_matrix(empty_matrix, collection_without_query, terms_set)
#        tf_matrix = idf(filled_matrix)
#        newlist = sentence_preview(tf_matrix, pre_preview)
        
        tfidf = rec.load_matrix()
        first_loading = process_text(pre_preview)
        prepared = prepare_query(tfidf, first_loading)
        try:
            output_result = document_preview(prepared, pre_preview)
        except ZeroDivisionError:
            os.remove(whole_path_string)
            return "No results were found. Return to the " + """<a href='/'>main page</a>"""
        os.remove(whole_path_string)

        return render_template('result.html', output_result = output_result)




@app.route('/processtext', methods=['GET', 'POST'])
def search():
    text = request.form['querytext']
    processed_text = text.lower()
    punctuation= '''!()-[]{};:'"\, <>./?@+#$%^&*_~'''
    stops = set(read_stopwords())
    doc_name = 'query'
    

    process_user_input = [processed_text]
    non_capital_list = []
    cleaned_list = []
    revised_list = []
    end_list = []
    dev_list = []
    words_count = {}

    
    for lines in process_user_input:
        non_capital_list.append(lines)
    for punc in non_capital_list:
        for character in punc.split():
            remove_punctuation = character.strip(punctuation)
            cleaned_list.append(remove_punctuation)
    for word in cleaned_list:
        if word not in stops:
            revised_list.append(word)
    for word in revised_list:
        for letter in word.split():
            short_word = letter.translate({ord(i): None for i in punctuation})
            end_list.append(short_word)
    for word in end_list:
        for letter in word.split():
            if len(letter) >= 3: 
                root_word = letter.rstrip("s")
                dev_list.append(root_word)
            else:
                dev_list.append(letter)
    for word in dev_list:
        for amount in word.split():
            words_count[word] = words_count.get(word, 0)+1
    
    my_dict = {doc_name:words_count}

    
    pre_preview = process_upload(process_user_input)
    tfidf = rec.load_matrix()
    first_loading = process_text(pre_preview)
    prepared = prepare_query(tfidf, first_loading)
    try:
        output_result = document_preview(prepared, pre_preview)
    except ZeroDivisionError:
        return "No results were found. Return to the " + """<a href='/'>main page</a>"""
    
    return render_template('result.html', output_result = output_result)

@app.route('/collection', methods=['GET', 'POST'])
def result_page():

    document_name = request.form['form_collection']
    cwd = os.getcwd()
    path_doc = cwd + '/collection/' + document_name
    path_copy = cwd + '/query.txt'
    shutil.copyfile(path_doc, path_copy)

    process_collection = process_text(path_copy)
    pre_preview = process_upload(path_copy)
#    collection_without_query = collectiondoc
#    collection_without_query.update(process_collection)
#    terms_set = unique_terms(collection_without_query)
#    empty_matrix = term_freq_matrix(collection_without_query, terms_set)
#    filled_matrix = add_matrix(empty_matrix, collection_without_query, terms_set)
#    tf_matrix = idf(filled_matrix)
#    newlist = sentence_preview(tf_matrix, pre_preview)

    collection_result = coll_x_coll(document_name)
    output_result = document_preview(collection_result, pre_preview)
    os.remove(path_copy)
    
    return render_template('result.html', output_result = output_result)

            

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER
    app.config['DEBUG'] = True
    app.config['SERVER_NAME'] = '127.0.0.1:5000'
    app.run()
        
    
    
    
    