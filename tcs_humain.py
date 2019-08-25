#!/usr/bin/env python
# coding: utf-8

# # importing the neccessary libraries

# In[ ]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# In[ ]:


import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text


# # reading the one sample of train data...for actual i need to read all the data but for sample code i took the sample data of 50000 rows of train data

# In[ ]:


train=pd.read_csv("E:/TCS_data/train.csv",nrows=50000)
#test=pd.read_csv("E:/TCS_data/Test.csv",nrows=100000)


# In[ ]:


print(train.columns) # getting the columns


# In[ ]:


train.head()
#print(train.columns)


# In[ ]:


#remove duplicate rows
train=train.drop_duplicates()


# In[ ]:


train.shape


# # preprocessing the data

# In[ ]:


#function to remove html tags from data
def remove_html(text):
    tag = re.compile(r'<.*?>')
    return tag.sub('', text)


# In[ ]:


#removing the html tags from Body column
for i in range(train.shape[0]):
    train.loc[i, 'Body'] = remove_html(train.loc[i,'Body'])


# In[ ]:


train.head()


# In[ ]:


def clean_words(sentence):#cleaning the text data
    sentence = sentence.lower()                # Converting to lowercase
    rem=re.compile(r'[^\w]')
    sentence=re.sub(rem,r' ', sentence)
    sentence = re.sub(r'[?$|!|\':|"|#]',r' ',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]\"\"]',r' ',sentence)#Removing Punctuations
    sentence=re.sub(r'\d+',' ',sentence)
    return sentence


# In[ ]:


#removing  unneccessary symbols from
for column in train:
    if(column=='Title' or column=='Body'):
        for i in range(train.shape[0]):
            train.loc[i, column] =clean_words(train.loc[i,column])
# for column in train:
#     index=0
#     if(column=='Title' or column=='Body'):
#         for sentence in train[column]:
#             cleaned_word=clean_words(sentence)
#             train[column][index]=cleaned_word  
#             index+=1


# In[ ]:


train.head()


# In[ ]:


#stop words 
stop_words=stopwords.words('english')


# In[ ]:


#tokenization
tokenized_doc_body=train['Body'].apply(lambda x:x.split())
tokenized_doc_title=train['Title'].apply(lambda x:x.split())
#remove stop_words
tokenized_doc_body=tokenized_doc_body.apply(lambda x:[item for item in x if item not in stop_words])
tokenized_doc_title=tokenized_doc_title.apply(lambda x:[item for item in x if item not in stop_words])
#de-tokenization
detokenized_doc_body=[]
detokenized_doc_title=[]
for i in range(train.shape[0]):
    t1=' '.join(tokenized_doc_body[i])
    t2=' '.join(tokenized_doc_title[i])
    detokenized_doc_body.append(t1)
    detokenized_doc_title.append(t2)
train['Body']= detokenized_doc_body
train['Title']= detokenized_doc_title
# def stop_words_remove(text):
#     tokenize_text=text.apply(lambda x:x.split())
#     final=tokenize_text.apply(lambda x:[item for item in x if item not in stop_words])
#     #de-tokenize
#     de_token=' '.join(final)
#     return de_token


# In[ ]:


train.head(7)
# for column in train:
#     if(column=='Title' or column=='Body'):
#         for i in range(train.shape[0]):
#             train.loc[i, column] =stop_words_remove(train.loc[i,column])


# # encoding the tags using multilabelbinarizer

# In[ ]:


#encode tags to multi-hot
train_tags=[]
for tag in train['Tags'].values:
    tags=[i for i in tag.split()]
    train_tags.append(tags)
#print(train_topics,'\n')
topic_encoder=MultiLabelBinarizer()
topic_encoded=topic_encoder.fit_transform(train_tags)


# In[ ]:


num_tags=len(topic_encoded[0])
print(num_tags)
print(topic_encoder.classes_)
print(topic_encoded[0])


# # to get document term matrix we use this function

# In[ ]:


#tokenize
class TextPreprocessor(object):
    def __init__(self,vocab_size):
        self._vocab_size=vocab_size
        self._tokenizer=None
    def create_tokenizer(self,text_list):
        tokenizer=text.Tokenizer(num_words=self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer=tokenizer
    def transform_text(self,text_list):
        text_matrix=self._tokenizer.texts_to_matrix(text_list)
        return text_matrix


# In[ ]:


train_body=train['Body'].values
train_title=train['Title'].values
# #test_text=test_data['Review Text'].values
# #test_title=test_data['Review Title'].values
# for i in range(train.shape[0]):
#     train[i]=train_text[i]+train_title[i]
# for i in range(test_text.shape[0]):
#     test_text[i]=test_text[i]+test_title[i]


# In[ ]:


vocab_size=500
processor=TextPreprocessor(vocab_size)
processor.create_tokenizer(train_body)
body_text=processor.transform_text(train_body)
body_title=processor.transform_text(train_tile)


# # horizontally stacking up the  body and title column term matrices

# In[ ]:


total_data=np.hstack(body_tile,body_text)


# In[ ]:


print(len(body_text[0]))
print(body_text[0])


# In[ ]:


print(body_text.shape) #  final size of the term matrix 


# # neural network

# In[ ]:


#create model
def create_model(num_topics,vocab_size):
    model=Sequential()
    model.add(Dense(128,input_shape=(vocab_size,),activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(num_topics,activation='softmax'))
    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model

model=create_model(num_tags,vocab_size)
model.summary()
    


# In[ ]:


#final model
model.fit(total_data,topic_encoded,epochs=10,batch_size=64,validation_split=0.1)


# # now our model is ready for the predictions.Let's test this model using test data

# # read the test data

# In[ ]:


test=pd.read_csv("E:/TCS_data/test.csv")


# In[ ]:


test.head() #top 5 rown of the test data


# In[ ]:


test.shape # shape of the test data


# In[ ]:


train=train.drop_duplicates() # drop_duplicates of test_data


# In[ ]:


for i in range(test.shape[0]):#remove html tags
    test.loc[i, 'Body'] = remove_html(test.loc[i,'Body'])


# In[ ]:


for column in test: # remove everything else which is scrap in the test dataset
    if(column=='Title' or column=='Body'):
        for i in range(test.shape[0]):
            test.loc[i, column] =clean_words(test.loc[i,column])


# In[ ]:


#tokenization
tokenized_doc_body_test=test['Body'].apply(lambda x:x.split())
tokenized_doc_title_test=test['Title'].apply(lambda x:x.split())
#remove stop_words
tokenized_doc_body_test=tokenized_doc_body_test.apply(lambda x:[item for item in x if item not in stop_words])
tokenized_doc_title_test=tokenized_doc_title_test.apply(lambda x:[item for item in x if item not in stop_words])
#de-tokenization
detokenized_doc_body_test=[]
detokenized_doc_title_test=[]
for i in range(test.shape[0]):
    t1=' '.join(tokenized_doc_body_test[i])
    t2=' '.join(tokenized_doc_title_test[i])
    detokenized_doc_body_test.append(t1)
    detokenized_doc_title_test.append(t2)
test['Body']= detokenized_doc_body_test
test['Title']= detokenized_doc_title_test


# In[ ]:


test_body=test['Body'].values
test_title=test['Title'].values#getting the values of the columns of body and title


# In[ ]:


#creating the term matrix for the test data
vocab_size=500
processor=TextPreprocessor(vocab_size)
processor.create_tokenizer(test_body)
body_text_test=processor.transform_text(test_body)
body_title_test=processor.transform_text(test_tile)


# In[ ]:


total_data_test=np.hstack(body_title_test,body_text_test)#horizontal stacking up of test body and title columns


# # finally we are at predictions step,the predicted_labels contains the tags for the test data.

# In[ ]:


predicted_labels=model.predict(total_data_test,batch_size=64)#predictions


# # this is my idea for the given problem statement using NLP and deep learning.My next task is to optimize the neural network for better accuracy.
