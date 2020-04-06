import spacy 
import neuralcoref
import re
import pandas as pd 
import string
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns


#load training and testing data
df = pd.read_csv('Entity_sentiment_trainV2.csv', index_col=0)
df_test = pd.read_csv('Entity_sentiment_testV2.csv')

#define preprocessing function: removing punctuation and lemmatizing
def preprocess(sent):
    pre = re.sub('\.', '\. ', sent)
    pre = pre.translate(str.maketrans('', '', string.punctuation))
    pre = pre.lower()
    pre = re.sub('  ', ' ', pre)
    pre = re.sub('  ', ' ', pre)
    pre = re.sub('  ', ' ', pre)
    doc = nlp(pre)
    lemmatized_doc = []
    for token in doc:
        lemmatized_token = token.lemma_
        if lemmatized_token == '-PRON-':
            lemmatized_doc.append(str(token))
        else:
            lemmatized_doc.append(lemmatized_token)
    lemmatized_string = " ".join(lemmatized_doc)
    return lemmatized_string

#define coreference resolution function
def resolve_coref_string(sent):
    doc = nlp(sent)
    resolved = doc._.coref_resolved
    return resolved

#define function to remove oov words
def remove_oov(sent): 
    doc = nlp(sent)
    new_sent = []
    for token in doc:
        if token.is_oov == True:
            None
        else:
            new_sent.append(str(token))
    new_sent_string = " ".join(new_sent)
    return new_sent_string

#apply these functions to both training and testing data 
df['preprocessed'] = df['Sentence'].apply(lambda x: preprocess(x))
df_test['preprocessed'] = df_test['Sentence'].apply(lambda x: preprocess(x))

df['Coreference_resolved'] = df['preprocessed'].progress_apply(lambda x: resolve_coref_string(x))
df_test['Coreference_resolved'] = df_test['preprocessed'].progress_apply(lambda x: resolve_coref_string(x))

df['No_oov'] = df['Coreference_resolved'].progress_apply(lambda x: remove_oov(x))
df_test['No_oov'] = df_test['Coreference_resolved'].progress_apply(lambda x: remove_oov(x))


#create training labels for model from preprocessed data
sentiments = df['Sentiment'].to_list()
sentiment_dict = {'positive': 1, 'negative':0}
sentiment_binary = [sentiment_dict.get(item, item) for item in sentiments]
labels = np.array(sentiment_binary)


#create training sentences and testing sentences from preprocessed data
corpus = df['No_oov'].to_list()
corpus_test = df_test['No_oov'].to_list()



#define number of cross-validation sets
kf = StratifiedKFold(n_splits=10)

#train model, predict labels for testing set, and create confusion matrix on validation set
totalsvm = 0
total_svm_conf_matrix = np.zeros((2, 2)) 

for train_index, validation_index in kf.split(corpus,labels):
    X_train = [corpus[i] for i in train_index]
    X_validation = [corpus[i] for i in validation_index]
    y_train, y_validation = labels[train_index], labels[validation_index]
    vectorizer = TfidfVectorizer(min_df=3, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    train_corpus_tf_idf = vectorizer.fit_transform(X_train)
    validation_corpus_tf_idf = vectorizer.transform(X_validation)
    test_tfidf = vectorizer.transform(corpus_test)
    
    model_svm = LinearSVC()
    model_svm.fit(train_corpus_tf_idf,y_train)
    result_svm = model_svm.predict(validation_corpus_tf_idf)
    test = model_svm.predict(test_tfidf)
    
    total_svm_conf_matrix = total_svm_conf_matrix + confusion_matrix(y_validation, result_svm)
    totalsvm = totalsvm+sum(y_validation==result_svm)


#convert testing labels back into words, add to testing df, and save to csv file
sent_dict_rev = {v: k for k, v in sentiment_dict.items()}
sentiment_binary_test = [sent_dict_rev.get(item, item) for item in test]
df_test['predicted_values'] = sentiment_binary_test

df_test.to_csv('Predicted_labels_testing_set.csv')


#display confusion matrix (as numpy array)
print(total_svm_conf_matrix)

#display precision, recall and f1-score
prec_rec_f1 = precision_recall_fscore_support(y_validation, result_svm, average='weighted')
print('Precision: ', prec_rec_f1[0], '\nRecall: ', prec_rec_f1[1], '\nF1-score: ', prec_rec_f1[2])

#create and save heatmap for easier visualisation of confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(total_svm_conf_matrix, annot=True, fmt='f',
            xticklabels=sentiment_dict.keys(), yticklabels=sentiment_dict.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_heatmap.png')
plt.show()









