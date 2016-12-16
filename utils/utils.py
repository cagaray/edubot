import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import nltk
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

#File names with moocs forum data.
fnames = ['BerkeleyX-BFV101.2x-3T2015-prod.mongo', 'BerkeleyX-BFV101x-T12015-prod.mongo', 'BerkeleyX-BJC.1x-3T2015-prod.mongo', 'BerkeleyX-ColWri_2.1x-3T2014-prod.mongo', 'BerkeleyX-ColWri2.1x-3T2015-prod.mongo', 'BerkeleyX-ColWri2.2x-1T2014-prod.mongo', 'BerkeleyX-ColWri.2.2x-1T2015-prod.mongo', 'BerkeleyX-ColWri2.3x-1T2014-prod.mongo', 'BerkeleyX-ColWri2.3x_2-1T2015-prod.mongo', 'BerkeleyX-ColWri3.1x-3T2014-prod.mongo', 'BerkeleyX-ColWri3.2x-1T2015-prod.mongo', 'BerkeleyX-ColWri3.3x-1T2015-prod.mongo', 'BerkeleyX-ColWri3.5x-1T2015-prod.mongo', 'BerkeleyX-ColWri.3.6x-3T2015-prod.mongo', 'BerkeleyX-ColWri.3.7x-3T2015-prod.mongo', 'BerkeleyX-ColWri.3.8x-3T2015-prod.mongo', 'BerkeleyX-ColWri.3.9x-3T2015-prod.mongo', 'BerkeleyX-CS100.1x-1T2015-prod.mongo', 'BerkeleyX-CS169.1x-3T2015-prod.mongo', 'BerkeleyX-CS_184.1x-3T2014-prod.mongo', 'BerkeleyX-CS188.1x-4-1T2015-prod.mongo', 'BerkeleyX-CS190.1x-1T2015-prod.mongo', 'BerkeleyX-EE40LX-1T2015-prod.mongo', 'BerkeleyX-EECS149.1x-2T2014-prod.mongo', 'BerkeleyX-GG101x-1T2014-prod.mongo', 'BerkeleyX-GG101x-2-1T2015-prod.mongo', 'BerkeleyX-GG101x-3T2015-prod.mongo', 'BerkeleyX-J4SC101-1T2015-prod.mongo', 'BerkeleyX-Stat_2.1x-1T2014-prod.mongo', 'BerkeleyX-Stat_2.2x-1T2014-prod.mongo', 'BerkeleyX-Stat_2.3x-2T2014-prod.mongo']

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

#Path to data folder.
data_path = '/research/edubot/repo/edubot/data/'

#Read all the files in the argument file_names to a json object. by default it'll take the array fnames, you should pass a new argument if needed.
def read_files_to_json(file_names = fnames):
    number_files = len(file_names)
    json_data = []
    count = 0
    for file_name in file_names:
        count = count + 1
        print('reading file %d of %d ' % (count, number_files))
        file_path = data_path + file_name
        remote_file = open(file_path)
        for line in remote_file:
            json_data.append(json.loads(line))
        remote_file.close()
    return json_data

#Read all the files in the argument file_names to a pandas object. by default it'll take the array fnames, you should pass a new argument if needed.
def read_files_to_pandas(file_names = fnames):
    number_files = len(file_names)
    json_data = []
    count = 0
    for file_name in file_names:
        count = count + 1
        print('reading file %d of %d ' % (count, number_files))
        file_path = data_path + file_name
        remote_file = open(file_path)
        for line in remote_file:
            json_data.append(json.loads(line))
        remote_file.close()
    return json_normalize(json_data)

#Read all data files and return a data frame with questions and their respective answers. A row will contain question information with sufix _q and answer information with sufix _a.
def get_qa_df(file_names = fnames):
    df = read_files_to_pandas(file_names)
    question_threads = df.loc[(df['_type'] == 'CommentThread') & (df['thread_type'] == 'question')][['title', 'body', 'author_id', 'course_id', '_id.$oid', 'parent_id.$oid']]
    print("%d question threads in the data" % len(question_threads))
    question_thread_ids = question_threads['_id.$oid'].tolist()
    answer_threads = df[df["comment_thread_id.$oid"].isin(question_thread_ids)][['title', 'body', 'author_id', 'course_id', '_id.$oid', 'comment_thread_id.$oid']]
    print("%d answer threads in the data" % len(answer_threads))
    all_qa = pd.merge(left=question_threads, right=answer_threads, left_on='_id.$oid', right_on='comment_thread_id.$oid', how='left', suffixes=('_q', '_a'))
    all_qa = all_qa.sort(['_id.$oid_q', 'comment_thread_id.$oid'])
    return all_qa

#Read all data files and return a list of questions and answers. The first row will be a question and the next rows will be its answers (could have 2 or more answers). The question will not be repeated before the second and subsequent answers.
def get_qa_list(file_names = fnames):
    df = read_files_to_pandas(file_names)
    question_threads = df.loc[(df['_type'] == 'CommentThread') & (df['thread_type'] == 'question')][['title', 'body', 'author_id', 'course_id', '_id.$oid', 'parent_id.$oid']]
    print("%d question threads in the data" % len(question_threads))
    question_thread_ids = question_threads['_id.$oid'].tolist()
    answer_threads = df[df["comment_thread_id.$oid"].isin(question_thread_ids)][['title', 'body', 'author_id', 'course_id', '_id.$oid', 'comment_thread_id.$oid']]
    print("%d answer threads in the data" % len(answer_threads))
    qa_list = []
    for index, row in question_threads.iterrows():
        qa_list.append(row['body'])
        q_id = row['_id.$oid']
        ans = answer_threads[answer_threads['comment_thread_id.$oid'] == q_id].sort(['_id.$oid'])
        for index2, row2 in ans.iterrows():
            qa_list.append(row2['body'])
    return qa_list

#Read all data files and return a list of questions and answers. For each par of rows, the first one will be a question and the next one will be its answer. If a question have 2 or more answers, the question will be repeated before adding the new answer.
def get_qa_list_qrepeated(file_names = fnames):
    all_qa = get_qa_df(fnames)
    qa_list = []
    for index, row in all_qa.iterrows():
        qa_list.append(row['body_q'])
        qa_list.append(row['body_a'])
    return(qa_list)

#Same as get_qa_list_qrepeated but containing only questions that have an answer.
def get_qa_list_qrepeated_notnull(file_names = fnames):
    all_qa = get_qa_df(fnames)
    all_qa = all_qa[all_qa['_id.$oid_a'].notnull()]
    qa_list = []
    for index, row in all_qa.iterrows():
        qa_list.append(row['body_q'])
        qa_list.append(row['body_a'])
    return(qa_list)

#Same as get_qa_list_qrepeated_notnull but the each row is a list of tokenized sentences.
def get_qa_list_qrepeated_notnull_tokenized(file_names = fnames):
    all_qa = get_qa_df(fnames)
    all_qa = all_qa[all_qa['_id.$oid_a'].notnull()]
    question_list = []
    answer_list = []
    for index, row in all_qa.iterrows():
        question_list.append(nltk.word_tokenize(row['body_q']))
        answer_list.append(nltk.word_tokenize(row['body_a']))
    return(question_list, answer_list)

#Read all data files and return a tuple with 2 lists, one for questions and one for answers. If a question has 2 or more answers the question list will have the question repeated. Contain only questions that have an answer.
def get_qa_lists(file_names = fnames):
    all_qa = get_qa_df(fnames)
    all_qa = all_qa[all_qa['_id.$oid_a'].notnull()]
    question_list = []
    answer_list = []
    for index, row in all_qa.iterrows():
        question_list.append(row['body_q'])
        answer_list.append(row['body_a'])
    return(question_list, answer_list)

#Tokenize, stem and remove stop-words of a sentence.
def tokenize_and_stem(text, tokenizer=tokenizer, stemmer=p_stemmer, stop_words=en_stop):
    return([stemmer.stem(word) for word in tokenizer.tokenize(text.lower()) if word not in stop_words])

#Tokenize and remove stop-words of a sentence.
def tokenize_only(text, tokenizer=tokenizer, stop_words=en_stop):
    return([word for word in tokenizer.tokenize(text.lower()) if word not in stop_words])