#!/usr/bin/env bash

# create and own the directories to store results locally
save_dir='/research/edubot/seq2seq_edubot/var/lib/tf_seq2seq_chatbot'
mkdir -p $save_dir'/data/'
mkdir -p $save_dir'/nn_models/'
mkdir -p $save_dir'/results/'
chown -R "$USER" $save_dir

# copy train and test data with proper naming
data_dir='tf_seq2seq_chatbot/data/train'
cp $data_dir'/discussion_forun_data.txt' $save_dir'/data/chat.in'
cp $data_dir'/discussion_forun_data_10k.txt' $save_dir'/data/chat_test.in'
