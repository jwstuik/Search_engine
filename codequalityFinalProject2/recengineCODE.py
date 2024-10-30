#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:32:02 2022

@author: jelletuik
"""


# Here the libraries that are used, are loaded.
import csv
import os
import math
import re
import glob
#import nltk


def read_stopwords():
    with open('stopwords.txt') as stops:
        content = stops.read()
        handle_backspace = content.split('\n')
        
    return handle_backspace

# Function to check an element and turn the value in
# a float if it is a number
def alterstring(s):
    try:
        s = float(s)
    except ValueError:
        pass
    return s

# Loop for csv files to turn them into a list.
def read_file_to_list():
    list_stories = []
    file = open("/Users/jelletuik/Downloads/development_material-3/stories_collection/stories.csv", newline='')
    content = csv.reader(file)
    for line in content:
        list_stories.append(line)
    file.close()
    return list_stories


# Loops over the list and uses the function alterstring()    
def list_to_float(nested_list):
    new_list = []
    for line in nested_list:
        second_list = []
        new_list.append(second_list)
        for column in line:
            alter_string = alterstring(column)
            second_list.append(alter_string)
    return new_list

read_list = list_to_float(read_file_to_list())

# This function is for calculating the document lentgh.
def doc_length(doc, matrix=read_list):
    processed_distance = []
    for line in matrix[1:]:
        term_weight = line[doc]
        exp_weight = term_weight ** 2
        processed_distance.append(exp_weight)
        corresponding_doc = matrix[0][doc]
    summarise_root = math.sqrt(sum(processed_distance))
    return list((corresponding_doc, summarise_root))
        
doc_matrix = list((doc_length(1,read_list), doc_length(2,read_list), doc_length(3, read_list), doc_length(4,read_list)))

# Here, the upper part of the cosine is calculated
def cos_sim_top(doc, query):
    products = []
    term_list = []
    query_list = []
    for row in read_list[1:]:
        chosen_doc = row[doc]
        term_list.append(chosen_doc)
    for row in read_list[1:]:
        chosen_query = row[query]
        query_list.append(chosen_query)
    for t1, q1 in zip(term_list, query_list):
        products.append(t1 * q1)
    return products

# when calling the function below, the function above is also executed
def cos_sim_below(doc, query):
    document_calc = doc_length(doc)
    query_calc = doc_length(query)
    calc_document = document_calc[1]
    doc_name = document_calc[0]
    calc_query = query_calc[1]
    calc_above = calc_document * calc_query
    acc_score = sum(cos_sim_top(doc, query))
    div_calc = acc_score / calc_above
    
    return [doc_name, div_calc]

prescale_1 = cos_sim_below(1,4)
prescale_2 = cos_sim_below(2,4)
prescale_3 = cos_sim_below(3,4)

# Here, the function checks to which scale the similarity belongs
def scale_matrix(doc_sim):
    pre_list = doc_sim
    if doc_sim[1] < 0.25:
        pre_list.append('low')
    if doc_sim[1] >= 0.25:
        if doc_sim[1] < 0.5:
            pre_list.append('medium')
    if doc_sim[1] >= 0.5:
        pre_list.append('high')
    return pre_list

scaled_doc1 = scale_matrix(prescale_1)
scaled_doc2 = scale_matrix(prescale_2)
scaled_doc3 = scale_matrix(prescale_3)

good_matrix = list((scaled_doc1,scaled_doc2,scaled_doc3))


# Function to sort the matrix 
def unranked_matrix(doc):
    unranked_list = []
    unranked_list.append(doc)
    return unranked_list

def sort(unranked_list):
    preranked_list = sorted(unranked_list, key = lambda x: x[1], reverse = True)
    return preranked_list

# Function to lowercase the text, do stemming
# remove stopwords and  punctuation.
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
        for amount in word.split():
            words_count[word] = words_count.get(word, 0)+1
    
    my_dict = {doc_name:words_count}
    return my_dict

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

# This function makes gives the ability to read all files in a specified
# directory while using the process_text() funtion.
def read_all_docs():
    files_dict = {}
    cwd = os.getcwd()
    path = cwd + '/collection/*.txt'
    for files in glob.glob(path):
        with open(files) as texts:
            temp_dict = process_text(files)
        files_dict.update(temp_dict)
    return files_dict



def load_matrix():
    
    
    # function that creates a set of unique words
    def unique_terms(nested_dict):
        set_dict = set()
        for doc in nested_dict:
            for word in nested_dict[doc]:
                set_dict.add(word)
        return set_dict
    
    test_collection = read_all_docs()    
    terms_set = unique_terms(test_collection)
    
    # Here, the empty matrix is created with only the 
    # the name of the documents and the words
    def term_freq_matrix(nested_dict, bow):
        columns = []
        matrix = []
        unique_list_terms = list(bow)
        for doc in nested_dict:
            columns.append(doc)
        columns.sort()
        columns.insert(0, 'term')
        matrix.append(columns)
        for term in range(len(unique_list_terms)):
            all_terms = unique_list_terms[term]
            matrix.append([all_terms])
        return matrix
    
    empty_matrix = term_freq_matrix(test_collection, terms_set)
    
    
    # Here the matrix is filled with term frequencies. 
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
    
    clear_matrix = add_matrix(empty_matrix, test_collection, terms_set)

    # The last preprocessing function that creates and calculcate the inverse
    # document frequencies.
    def idf(base_matrix=clear_matrix):
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
    
            
    clear_matrix_2 = idf(clear_matrix)
    return clear_matrix_2


well_query1 = '/Users/jelletuik/Desktop/acm_apple_query_test.txt'


processed_well_query = process_upload(well_query1)
nogeensverwerken = process_text(processed_well_query)


cwd = os.getcwd()

testquery = ['gekomen', 'Shell', 'vallen','dividend', 'zonvakantie', 'Apple', 'failliet', 'gevaar', 'maatregelen', 'houden', 'gepubliceerd']
#proc_test_query = process_text(testquery)
#query_term_set = term_freq_matrix(proc_test_query, terms_set)
#tf_query = add_matrix(query_term_set, proc_test_query, terms_set)

#inside_test = read_all_docs()
#inside_q_test = process_text(testquery) 
#inside_test.update(nogeensverwerken)

#inside_terms_set = unique_terms(inside_test)
#lege_matrix = term_freq_matrix(inside_test, inside_terms_set)
#gevulde_matrix = add_matrix(lege_matrix, inside_test, inside_terms_set)  
#inside_idf_matrix = idf(gevulde_matrix)




def square(number):
    return (number * number)

def column_sum(document):
    return [sum(i) for i in zip(*document)]

def document_length(collection):
    squared_matrix = []
    summed_matrix = []
    columns = collection[0]
    squared_matrix.append(columns)
    for document in collection[1:]:
        rows = list(document[:1])
        for words in document[1:]:
            square_values = square(words)
            rows.append(square_values)
        squared_matrix.append(rows)
        
    
    for lines in squared_matrix[1:]:
        subvalue = []
        for value in lines[1:]:
            subvalue.append(value)
        summed_matrix.append(subvalue)
        
    total_matrix = column_sum(summed_matrix)
    final_doc_length = []
    final_doc_length.append(collection[0][1:])
    for total in total_matrix:
        length = math.sqrt(total)
        final_doc_length.append([length])
        
                
    return final_doc_length




    
        
        
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
            

def cosine(collection):
    run_document_length = document_length(collection)
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



def search_sentence(document):
    file = open(document, 'r')
    strings = re.findall(r"\w+|\W+", file.read(), flags =re.DOTALL)
    return strings
        



def sentence_preview(collection, query):
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
            
tfidf = load_matrix()


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

#prepared = prepare_query(tfidf, nogeensverwerken)
#resultaat = sentence_preview(prepared, processed_well_query)

#resultaat2 = resultaat[0]
#print(resultaat2)
#print(resultaat2[5][resultaat2[5].index(resultaat2[4])+len(resultaat2[4]):])
            
            


def document_preview(query):
    cwd = os.getcwd()
    path_doc = cwd + '/collection/'
    result = []
    for file in os.listdir(path_doc):
        current_path = os.path.join(path_doc, file)
        if current_path.endswith('.txt'):
            
            with open(current_path, 'r') as file:
                content = file.read()
                remove_space = content.split('\n')
                #doc_sentence = []
            #sentence = []
            for document in remove_space:
                doc_sentence = []
                for term in query:
                    sentence = []
         
                    if term in document:

                        position = document.index(term)
                        preview = document[position-25:position+25]
                        if position == 0:
                            continue
                        else:
                            doc_sentence.append(os.path.basename(current_path))
                            sentence.append(preview)
                        
                            doc_sentence.append(sentence)
                        result.append(doc_sentence)


    
    return result





def old_preview(collection, query):
    cwd = os.getcwd()
    path_doc = cwd + '/collection/'
    whole_path = glob.glob(path_doc + '*.txt')
    sentence_result = []
    ranked_mtx = show_output(collection)
    preview_list = ranked_mtx[1:]
    newlist = preview_list

    
    relevant_documents = []
    
    for hits in preview_list:
        relevant_documents.append(hits[0])
        
    for doc in range(len(relevant_documents)):
        with open(relevant_documents[doc]) as sentence:
            temp = sentence.read()
            temp1 = temp.split('\n')
            words = re.findall(r"\w+|\W+",temp)
            for term in range(len(query)):
                if query[term] in words:
                    keyword = words.index(query[term])
                    if keyword != 0:
                        preview = words[keyword-10:keyword+10]
                        prev_string = "".join(preview)
                        newlist[doc].insert(3, prev_string)
                        newlist[doc].insert(4, words[keyword])   
        
        
    return newlist


def coll_x_coll(document):
    tfidf = load_matrix()
    tfidf_col = tfidf
    position = tfidf[0].index(document)
    tfidf_col[0].append(document)
    for doc in range(1,len(tfidf)):
        tfidf_col[doc].append(tfidf_col[doc][position])
        
    return tfidf_col

print(show_output(coll_x_coll('document10.txt')))
        
            
        
            

        
        
        
    
