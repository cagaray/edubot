# edubot
A Sequence-to-Sequence approach to finding relevant answers to questions in MOOC forums.

All of our code was writen in Python using Jupyter notebooks and can be reproduced giving the necesary data. The dataset is private and is not included with this code.

Explain this exercise.

## Code organization

### doc2vec
This folder contains the necesary code to run our Doc2Vec experimentation. We used Doc2Vec to, given a new question, find similar questions in our dataset and the answers to those questions.

### lda
This folder contains the necesary code to run our LDA topic modelling experimentation. We used LDA topic modelling to infer the different topic of all the questions in our dataset.

### seq2seq_edubot
This folder contains our attempt to use Seq2Seq to generate new answers to new questions based on our existing dataset. This code was based on the work of Nicolas Ivanov and the original code can be found here: https://github.com/nicolas-ivanov/tf_seq2seq_chatbot

To run this model please edit the files _setup.sh_ and _tf_seq2seq_chatbot/configs/_ with the appropiate values.
To train the model run _train.py_.
The model is already trained (8,000 steps to achieve perplexity of 17) and can be tested with the script _chat.py_.

### utils
This folder contains functions that are used across all scripts (reading data, generate question lists, etc.). Please edit the appropiate paths to the data files and the file names.

## Dataset
All of our work was made with a private dataset of questions and answers for several MOOC courses from edX. This data is not included and in order to run these scripts there should be added in a folder named _data_ on the root of the project.
