#!/usr/bin/env python
# coding: utf-8

# # importing the neccessary libraries

# In[1]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# In[2]:


import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")


# In[3]:


import tensorflow as tf
from keras.optimizers import *
from keras import regularizers
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text


# In[4]:


import pickle


# # reading the data

# In[5]:


train=pd.read_csv("E:/TCS_data/train.csv",nrows=150000)
#test=pd.read_csv("E:/TCS_data/Test.csv",nrows=100000)


# In[6]:


print(train.columns) # getting the columns


# In[7]:


train.head()
#print(train.columns)


# In[8]:


train.shape


# # preprocessing the data

# In[9]:


#function to remove html tags from data
def remove_html(text):
    tag = re.compile(r'<.*?>')
    return tag.sub('', text)


# In[10]:


#removing the html tags from Body column
for i in range(train.shape[0]):
    train.loc[i, 'Body'] = remove_html(train.loc[i,'Body'])


# In[11]:


train.head()


# In[12]:


def clean_words(sentence):#cleaning the text data
    sentence = sentence.lower()                # Converting to lowercase
    rem=re.compile(r'[^\w]')
    sentence=re.sub(rem,r' ', sentence)
    sentence=re.sub(r'(?:^| )\w(?:$| )', r' ', sentence)#removing single characters
    sentence = re.sub(r'[?$|!|\':|"|#]',r' ',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]\"\"]',r' ',sentence)#Removing Punctuations
    sentence=re.sub(r'\d+',' ',sentence)
    return sentence


# In[13]:


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


# In[14]:


train.head()


# In[15]:


#stop words 
stop_words=stopwords.words('english')


# In[16]:


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


# In[17]:


train.head(5)
# for column in train:
#     if(column=='Title' or column=='Body'):
#         for i in range(train.shape[0]):
#             train.loc[i, column] =stop_words_remove(train.loc[i,column])


# # encoding the tags using multilabelbinarizer

# In[18]:


train=shuffle(train,random_state=5)


# In[19]:


train.head()


# In[20]:


#encode tags to multi-hot
train_tags=[]
for tag in train['Tags'].values:
    tags=[i for i in tag.split()]
    train_tags.append(tags)
#print(train_topics,'\n')
topic_encoder=MultiLabelBinarizer()
topic_encoded=topic_encoder.fit_transform(train_tags)


# In[21]:


num_tags=len(topic_encoded[0])
print(num_tags)
print(topic_encoder.classes_)
print(topic_encoded[0])


# # to get document term matrix we use this function

# In[22]:


#tokenize
class TextPreprocessor(object):
    def __init__(self,vocab_size):
        self._vocab_size=vocab_size
        self._tokenizer=None
    def create_tokenizer(self,text_list):
        tokenizer=text.Tokenizer(num_words=self._vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(text_list)
        self._tokenizer=tokenizer
    def transform_text(self,text_list):
        text_matrix=self._tokenizer.texts_to_matrix(text_list)
        return text_matrix


# In[23]:


train_body=train['Body'].values
#train_title=train['Title'].values
# #test_text=test_data['Review Text'].values
# #test_title=test_data['Review Title'].values
# for i in range(train.shape[0]):
#     train[i]=train_text[i]+train_title[i]
# for i in range(test_text.shape[0]):
#     test_text[i]=test_text[i]+test_title[i]


# In[24]:


vocab_size=600
processor=TextPreprocessor(vocab_size)
processor.create_tokenizer(train_body)
body_text=processor.transform_text(train_body)
#body_title=processor.transform_text(train_title)


# In[25]:


with open('./processor_state_f1.pkl','wb') as f:
    pickle.dump(processor,f)


# # horizontally stacking up the  body and title column term matrices

# In[26]:


#total_data=np.hstack(body_tile,body_text)


# In[27]:


print(len(body_text[0]))
print(body_text[0])


# In[28]:


print(body_text.shape) #  final size of the term matrix 


# In[29]:


words=processor._tokenizer.word_index
word_lookup=list()
for i in words.keys():
    word_lookup.append(i)
print(len(word_lookup))


# # neural network

# In[30]:


#load_processor=pickle.load(open('processor_state_f.pkl','rb'))


# In[30]:


#create model
#def create_model(num_topics,vocab_size):
model=Sequential()
model.add(Dense(128,input_shape=(vocab_size,),activation='relu'))
model.add(Dense(50,activation='relu' ))
model.add(Dropout(0.5))
#model.add(Dense(32,activation='relu'))
model.add(Dense(30,activation='relu' ))
model.add(Dropout(0.5))
model.add(Dense(num_tags,activation='sigmoid'))
#adam=adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model=create_model(num_tags,vocab_size)
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[31]:


model.fit(body_text,topic_encoded,epochs=31,batch_size=128,validation_split=0.1)


# In[48]:


#pickel_model=pickle.dump()
with open('./model_state_31.pkl','wb') as f:
    pickle.dump(model,f)


# In[33]:


## for 0.5 drop out accuracy are for 
#1.epoch 25 71.64 40 99.73% 32 82%
##for 0.6 drop out accuracy are for
#1.epoch29 74.9% 31 83.43% 35  98.68%


# # now our model is ready for the predictions.Let's test this model using test data

# # read the test data

# In[32]:


test=pd.read_csv("E:/TCS_data/test.csv")


# In[33]:


test.head() #top 5 rown of the test data


# In[34]:


test.shape # shape of the test data


# In[35]:


train=train.drop_duplicates() # drop_duplicates of test_data


# In[36]:


for i in range(test.shape[0]):#remove html tags
    test.loc[i, 'Body'] = remove_html(test.loc[i,'Body'])


# In[37]:


for column in test: # remove everything else which is scrap in the test dataset
    if(column=='Title' or column=='Body'):
        for i in range(test.shape[0]):
            test.loc[i, column] =clean_words(test.loc[i,column])


# In[38]:


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
#test['Title']= detokenized_doc_title_test


# In[41]:


test.head()


# In[43]:


test=shuffle(test,random_state=505)


# In[44]:


test.head()


# In[39]:


test_body=test['Body'].values
#test_title=test['Title'].values#getting the values of the columns of body and title


# In[45]:


#creating the term matrix for the test data
vocab_size=600
processor=TextPreprocessor(vocab_size)
processor.create_tokenizer(test_body)
body_text_test=processor.transform_text(test_body)
#body_title_test=processor.transform_text(test_tile)


# In[ ]:


#total_data_test=np.hstack(body_title_test,body_text_test)#horizontal stacking up of test body and title columns


# # finally we are at predictions step,the predicted_labels contains the tags for the test data.

# In[46]:


predicted_labels=model.predict(body_text_test,batch_size=128)#predictions


# In[47]:


predicted_labels


# # this is my idea for the given problem statement using NLP and deep learning.
