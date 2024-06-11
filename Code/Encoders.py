from NewsContent import *
from UserContent import *
from preprocessing import *
from models import *
from utils import *

import os
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


class PositionEmbedding(nn.Module):
    def _init_(self, sequence_length, embedding_dim, initializer='glorot_uniform'):
        super(PositionEmbedding, self)._init_()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.position_embeddings = nn.Parameter(torch.Tensor(sequence_length, embedding_dim))
        self.reset_parameters(initializer)

    def reset_parameters(self, initializer):
        if initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.position_embeddings)
        # Add more initializers as needed based on your requirements

    def forward(self, x, start_index=0):
        # Adjust the position embeddings to match the input size, handling varying sequence lengths
        max_length = x.size(1)
        position_embeddings = self.position_embeddings[start_index:start_index + max_length, :]
        
        # Ensure position embedding is broadcastable over the batch size
        # x.shape is (batch_size, sequence_length, embedding_dim)
        position_embeddings = position_embeddings.unsqueeze(0).expand_as(x)
        return x + position_embeddings

# def get_doc_encoder(config,text_length,embedding_layer):
#     news_encoder = config['news_encoder_name']
#     # sentence_input = Input(shape=(1,), dtype = 'int32')
#     sentence_input = torch.empty((text_length,), dtype=torch.int32)
#     embedded_sequences = embedding_layer(sentence_input)


#     # d_et = Dropout(0.2)(embedded_sequences)
#     d_et=F.dropout(embedded_sequences, p= 0.2)

#     if news_encoder=='CNN':
#         # l_cnnt = Conv1D(400,kernel_size=3,activation='relu')(d_et)
#         l_cnnt = F.conv1d(d_et, torch.randn(400, text_length, 3))
#         l_cnnt = F.relu(l_cnnt)
        
#     elif news_encoder=='SelfAtt':
#         # l_cnnt =Attention(20,20)([d_et,d_et,d_et])
#         l_cnnt = Attention(20, 20)(d_et,d_et,d_et)

#     elif news_encoder=='SelfAttPE':
#         # d_et = PositionEmbedding(title_length,300)(d_et) keras.layers.PositionEmbedding
#         d_et = ???

#         # l_cnnt =Attention(20,20)([d_et,d_et,d_et])
#         l_cnnt = Attention(20, 20)(d_et,d_et,d_et)
#     # d_ct=Dropout(0.2)(l_cnnt)
#     d_ct = F.dropout(l_cnnt, p= 0.2)
#     l_att = AttentivePooling(text_length,400)(d_ct)
#     sentEncodert = Model(sentence_input, l_att) #keras.models
#     return sentEncodert

class Doc_encoder(nn.Module):
    def __init__(self, config, text_length, embedding_layer):
        self.news_encoder = config['news_encoder_name']
        self.embedding_layer = embedding_layer
        self.dropout = nn.Dropout(p=0.2)
        # is it size 1 as in one document?
        self.conv = nn.Conv1d(1, 400, 3)
        self.relu = nn.ReLU()
        self.attention = Attention(20,20)
        self.attentivePool = AttentivePooling(text_length, 400)


    def forward(self, x):
        x = self.embedding_layer(x) 
        x = self.dropout(x)

        if self.news_encoder=='CNN':
            x = self.conv(x)
            x = self.relu(x)
            
        elif self.news_encoder=='SelfAtt':
            x = self.attention(x, x, x)

        elif self.news_encoder=='SelfAttPE':
            # d_et = PositionEmbedding(title_length,300)(d_et) keras.layers.PositionEmbedding
            # x = PositionEmbedding(self.tit)
            # x = self.attention(x, x, x)
            ...
        
        x = self.dropout(x)
        x = self.attentivePool(x)
        return x



# def get_vert_encoder(config,vert_num,):
#     vert_input = Input(shape=(1,))
#     embedding_layer = Embedding(vert_num+1, 400,trainable=True)
#     vert_emb = embedding_layer(vert_input)
#     vert_emb = keras.layers.Reshape((400,))(vert_emb)
#     vert_emb = Dropout(0.2)(vert_emb)
#     model = Model(vert_input,vert_emb)
#     return model
        
class Vert_encoder(nn.Module):
    def __init__(self, config, vert_num):
        self.news_encoder = config['news_encoder_name']
        self.embedding_layer = nn.Embedding(vert_num+1, 400) # keras.layers.embedding
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.embedding_layer(x) 
        x = x.view(400, -1) #reshape ?
        x = self.dropout(x)
        return x

# def get_news_encoder(config,vert_num,subvert_num,word_num,word_embedding_matrix,entity_embedding_matrix):
#     LengthTable = {'title':config['title_length'],
#                    'body':config['body_length'],
#                    'vert':1,'subvert':1,
#                    'entity':config['max_entity_num']}
#     input_length = 0
#     PositionTable = {}
#     for v in config['attrs']:
#         PositionTable[v] = (input_length,input_length+LengthTable[v])
#         input_length += LengthTable[v]
#     print(PositionTable)
#     word_embedding_layer = Embedding(word_num+1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],trainable=True)

#     news_input = Input((input_length,),dtype='int32')
    
#     title_vec = None
#     body_vec = None
#     vert_vec = None
#     subvert_vec = None
#     entity_vec = None
    
#     if 'title' in config['attrs']:
#         title_input = keras.layers.Lambda(lambda x:x[:,PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
#         title_encoder = get_doc_encoder(config,LengthTable['title'],word_embedding_layer)
#         title_vec = title_encoder(title_input)
        
#     if 'body' in config['attrs']:
#         body_input = keras.layers.Lambda(lambda x:x[:,PositionTable['body'][0]:PositionTable['body'][1]])(news_input)
#         body_encoder = get_doc_encoder(config,LengthTable['body'],word_embedding_layer)
#         body_vec = body_encoder(body_input)

#     if 'vert' in config['attrs']:

#         vert_input = keras.layers.Lambda(lambda x:x[:,PositionTable['vert'][0]:PositionTable['vert'][1]])(news_input)
#         vert_encoder = get_vert_encoder(config,vert_num)
#         vert_vec = vert_encoder(vert_input)
    
#     if 'subvert' in config['attrs']:
#         subvert_input = keras.layers.Lambda(lambda x:x[:,PositionTable['subvert'][0]:PositionTable['subvert'][1]])(news_input)
#         subvert_encoder = get_vert_encoder(config,subvert_num)
#         subvert_vec = subvert_encoder(subvert_input)
    
#     if 'entity' in config['attrs']:
#         entity_input = keras.layers.Lambda(lambda x:x[:,PositionTable['entity'][0]:PositionTable['entity'][1]])(news_input)
#         entity_embedding_layer = Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],trainable=False)
#         entity_emb = entity_embedding_layer(entity_input)
#         entity_vecs = Attention(20,20)([entity_emb,entity_emb,entity_emb])
#         entity_vec = AttentivePooling(LengthTable['entity'],400)(entity_vecs)
        
#     vec_Table = {'title':title_vec,'body':body_vec,'vert':vert_vec,'subvert':subvert_vec,'entity':entity_vec}
#     feature = []
#     for attr in config['attrs']:
#         feature.append(vec_Table[attr])
#     if len(feature)==1:
#         news_vec = feature[0]
#     else:
#         for i in range(len(feature)):
#             feature[i] = keras.layers.Reshape((1,400))(feature[i])
#         news_vecs = keras.layers.Concatenate(axis=1)(feature)
#         news_vec = AttentivePooling(len(config['attrs']),400)(news_vecs)
#     model = Model(news_input,news_vec)
#     return model

class News_encoder(nn.Module):
    def __init__(self, config, vert_num, subvert_num, word_num, word_embedding_matrix, entity_embedding_matrix):
        self.config = config
        self.vert_num = vert_num
        self.subvert_num = subvert_num
        self.word_num = word_num
        self.word_embedding_matrix = word_embedding_matrix
        self.entity_embedding_matrix = entity_embedding_matrix
        self.LengthTable = {'title':config['title_length'],
                   'body':config['body_length'],
                   'vert':1,'subvert':1,
                   'entity':config['max_entity_num']}
        self.input_length = 0
        self.PositionTable = {}
        for v in config['attrs']:
            self.PositionTable[v] = (self.input_length, self.input_length + self.LengthTable[v])
            self.input_length += self.LengthTable[v]
        
        self.word_embedding_layer = nn.Embedding(self.word_num + 1, word_embedding_matrix.shape[1], _weight=self.word_embedding_matrix) # keras.layers.embedding TODO Check if weights are a tensor
        self.entity_embedding_layer = nn.Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1], _freeze = True) #Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],trainable=False)
        self.attention = Attention(20,20)
        self.attentive_pool = AttentivePooling(self.LengthTable['entity'], 400)
        
        self.title_vec = None
        self.body_vec = None
        self.vert_vec = None
        self.subvert_vec = None
        self.entity_vec = None

    def forward(self, x):

        if 'title' in self.config['attrs']:
            title_input = lambda xi: xi[:, self.PositionTable['title'][0] : self.PositionTable['title'][1]] in x
            title_encoder = Doc_encoder(self.config, self.LengthTable['title'], self.word_embedding_layer)
            title_vec = title_encoder(title_input)
        
        if 'body' in self.config['attrs']:
            body_input = lambda xi: xi[:, self.PositionTable['body'][0] : self.PositionTable['body'][1]] in x
            body_encoder = Doc_encoder(self.config, self.LengthTable['body'], self.word_embedding_layer)
            body_vec = body_encoder(body_input)

        if 'vert' in self.config['attrs']:
            vert_input = lambda xi: xi[:, self.PositionTable['vert'][0] : self.PositionTable['vert'][1]] in x
            vert_encoder = Vert_encoder(self.config, self.vert_num)
            vert_vec = vert_encoder(vert_input)
        
        if 'subvert' in self.config['attrs']:
            subvert_input = lambda xi: xi[:, self.PositionTable['subvert'][0] : self.PositionTable['subvert'][1]] in x
            subvert_encoder = Vert_encoder(self.config, self.subvert_num)
            subvert_vec = subvert_encoder(subvert_input)
        
        if 'entity' in self.config['attrs']:
            entity_input = lambda xi: xi[:, self.PositionTable['entity'][0] : self.PositionTable['entity'][1]] in x
            entity_emb = self.entity_embedding_layer(entity_input)
            entity_vecs = self.attention(entity_emb,entity_emb,entity_emb)
            entity_vec = self.attentive_pool(entity_vecs)

        vec_Table = {'title':title_vec,'body':body_vec,'vert':vert_vec,'subvert':subvert_vec,'entity':entity_vec}
        feature = []
        for attr in self.config['attrs']:
            feature.append(vec_Table[attr])

        if len(feature)==1:
            news_vec = feature[0]
        else:
            for i in range(len(feature)):
                feature[i] = torch.reshape(feature(i), (1, 400)) #keras.layers.Reshape((1,400))(feature[i])    
            news_vecs = torch.cat(feature, dim = 1)   # keras.layers.Concatenate(axis=1)(feature)
            news_vec = AttentivePooling(len(self.config['attrs']), 400)(news_vecs)
    

        return news_vec

# This function is never used   
# def create_model(config,News,word_embedding_matrix,entity_embedding_matrix):
#     max_clicked_news = config['max_clicked_news']
        
#     # news_encoder = get_news_encoder(config,len(News.category_dict),len(News.subcategory_dict),len(News.word_dict),word_embedding_matrix,entity_embedding_matrix)
#     news_encoder = News_encoder(config,len(News.category_dict),len(News.subcategory_dict),len(News.word_dict),word_embedding_matrix,entity_embedding_matrix)
#     news_input_length = int(news_encoder.input.shape[1])
#     # print(news_input_length)
#     clicked_input = Input(shape=(max_clicked_news, news_input_length,), dtype='int32')
#     # print(clicked_input.shape)
#     user_vecs = TimeDistributed(news_encoder)(clicked_input)

#     if config['user_encoder_name'] =='SelfAtt':
#         user_vecs = Attention(20,20)([user_vecs,user_vecs,user_vecs])
#         user_vecs = Dropout(0.2)(user_vecs)
#         user_vec = AttentivePooling(max_clicked_news,400)(user_vecs)
#     elif config['user_encoder_name'] == 'Att':
#         user_vecs = Dropout(0.2)(user_vecs)
#         user_vec = AttentivePooling(max_clicked_news,400)(user_vecs)
#     elif config['user_encoder_name'] == 'GRU':
#         user_vecs = Dropout(0.2)(user_vecs)
#         user_vec = GRU(400,activation='tanh')(user_vecs)
        
#     candidates = keras.Input((1+config['npratio'],news_input_length,), dtype='int32')
#     candidate_vecs = TimeDistributed(news_encoder)(candidates)
#     score = keras.layers.Dot(axes=-1)([user_vec,candidate_vecs])
#     logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(score)

#     model = Model([candidates,clicked_input], [logits])

#     model.compile(loss=['categorical_crossentropy'],
#                   optimizer=Adam(lr=0.0001), 
#                   metrics=['acc'])

#     user_encoder = Model([clicked_input],user_vec)
    
#     return model,user_encoder,news_encoder



# def get_news_encoder_co1(config,vert_num,subvert_num,word_num,word_embedding_matrix,entity_embedding_matrix):
#     LengthTable = {'title':config['title_length'],
#                    'vert':1,'subvert':1,
#                    'entity':config['max_entity_num']}
#     input_length = 0
#     PositionTable = {}
#     for v in config['attrs']:
#         PositionTable[v] = (input_length,input_length+LengthTable[v])
#         input_length += LengthTable[v]
#     print(PositionTable)
#     word_embedding_layer = Embedding(word_num+1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],trainable=True)

#     news_input = Input((input_length,),dtype='int32')
    
#     vert_input = keras.layers.Lambda(lambda x:x[:,PositionTable['vert'][0]:PositionTable['vert'][1]])(news_input)
#     vert_embedding_layer = Embedding(vert_num+1, 200,trainable=True)
#     vert_emb = vert_embedding_layer(vert_input)
#     vert_emb = keras.layers.Reshape((200,))(vert_emb)
#     vert_vec = Dropout(0.2)(vert_emb)
    
#     title_input = keras.layers.Lambda(lambda x:x[:,PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
#     title_emb = word_embedding_layer(title_input)
#     title_emb = Dropout(0.2)(title_emb)
        
    
#     entity_input = keras.layers.Lambda(lambda x:x[:,PositionTable['entity'][0]:PositionTable['entity'][1]])(news_input)
#     entity_embedding_layer = Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],trainable=True)
#     entity_emb = entity_embedding_layer(entity_input)
    
#     title_co_emb = Attention(5,40)([title_emb,entity_emb,entity_emb])
#     entity_co_emb = Attention(5,40)([entity_emb,title_emb,title_emb])
    
#     title_vecs = Attention(20,20)([title_emb,title_emb,title_emb])
#     title_vecs = keras.layers.Concatenate(axis=-1)([title_vecs,title_co_emb])
#     title_vecs = Dense(400)(title_vecs)
#     title_vecs = Dropout(0.2)(title_vecs)
#     title_vec = AttentivePooling(config['title_length'],400)(title_vecs)
    
#     entity_vecs = Attention(20,20)([entity_emb,entity_emb,entity_emb])
#     entity_vecs = keras.layers.Concatenate(axis=-1)([entity_vecs,entity_co_emb])
#     entity_vecs = Dense(400)(entity_vecs)
#     entity_vecs = Dropout(0.2)(entity_vecs)
#     entity_vec = AttentivePooling(LengthTable['entity'],400)(entity_vecs)
                
#     feature = [title_vec,entity_vec,vert_vec]

#     news_vec = keras.layers.Concatenate(axis=-1)(feature)
#     news_vec = Dense(400)(news_vec)
#     model = Model(news_input,news_vec)
#     return model

class News_encoder_co1(nn.Module):
    def __init__(self, config, vert_num, subvert_num, word_num, word_embedding_matrix, entity_embedding_matrix):
        self.config = config
        self.vert_num = vert_num
        self.subvert_num = subvert_num
        self.word_num = word_num
        self.word_embedding_matrix = word_embedding_matrix
        self.entity_embedding_matrix = entity_embedding_matrix
        self.LengthTable = {'title':config['title_length'],
                   'vert':1,'subvert':1,
                   'entity':config['max_entity_num']}
        self.input_length = 0
        self.PositionTable = {}
        for v in config['attrs']:
            self.PositionTable[v] = (self.input_length, self.input_length + self.LengthTable[v])
            self.input_length += self.LengthTable[v]
        
        self.word_embedding_layer = nn.Embedding(self.word_num+1, self.word_embedding_matrix.shape[1], _weight=self.word_embedding_matrix) # keras.layers.embedding
        self.vert_embedding_layer = nn.Embedding(self.vert_num + 1, 200) # Embedding(vert_num+1, 200,trainable=True) TODO trainable y/n
        self.entity_embedding_layer = nn.Embedding(self.entity_embedding_matrix.shape[0], self.entity_embedding_matrix.shape[1]) # Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],trainable=True)
        self.attention = Attention(20,20)
        self.attention2 = Attention(5,40)
        self.attentive_pool = AttentivePooling(self.LengthTable['entity'], 400)
        self.attentive_pool2 = AttentivePooling(self.config['title_length'], 400)
        self.dropout = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(20*20*5*40,400)
        self.fc2 = nn.Linear(20*20*5*40,400)
        self.fc3 = ... #nn.Linear(???,400) TODO choose random and see?
        
        self.title_vec = None
        self.body_vec = None
        self.vert_vec = None
        self.subvert_vec = None
        self.entity_vec = None

    def forward(self, x):

        vert_input = lambda xi: xi[:, self.PositionTable['vert'][0] : self.PositionTable['vert'][1]] in x
        vert_emb = self.vert_embedding_layer(vert_input)
        vert_emb = vert_emb.view(200,-1)
        vert_vec = self.dropout(vert_emb)

        title_input = lambda xi: xi[:, self.PositionTable['title'][0] : self.PositionTable['title'][1]] in x
        title_emb = self.word_embedding_layer(title_input)
        title_emb = self.dropout(title_emb)

        entity_input = lambda xi: xi[:, self.PositionTable['entity'][0] : self.PositionTable['entity'][1]] in x
        entity_emb = self.entity_embedding_layer(entity_input)

        title_co_emb = self.attention2(title_emb,entity_emb,entity_emb)
        entity_co_emb = self.attention2(entity_emb,title_emb,title_emb)
        
        title_vecs = self.attention(title_emb, title_emb, title_emb)
        title_vecs = torch.cat((title_vecs, title_co_emb), dim = -1)
        title_vecs = self.fc1(title_vecs)
        title_vecs = self.dropout(title_vecs)
        title_vec = self.attentive_pool2(title_vecs)

        entity_vecs = self.attention(entity_emb, entity_emb, entity_emb)
        entity_vecs = torch.cat((entity_vecs, entity_co_emb), dim = -1)
        entity_vecs = self.fc2(entity_vecs)
        entity_vecs = self.dropout(entity_vecs)
        entity_vec = self.attentive_pool(entity_vecs)

        feature = [title_vec, entity_vec, vert_vec]

        news_vec = torch.cat(feature, dim = -1)
        news_vec = self.fc3(news_vec)
        return news_vec
        

    


# Never used
# def get_news_encoder_co2(config,vert_num,subvert_num,word_num,word_embedding_matrix,entity_embedding_matrix):
#     LengthTable = {'title':config['title_length'],
#                    'vert':1,'subvert':1,
#                    'entity':config['max_entity_num']}
#     input_length = 0
#     PositionTable = {}
#     for v in config['attrs']:
#         PositionTable[v] = (input_length,input_length+LengthTable[v])
#         input_length += LengthTable[v]
#     print(PositionTable)
#     word_embedding_layer = Embedding(word_num+1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],trainable=True)

#     news_input = Input((input_length,),dtype='int32')
    
#     vert_input = keras.layers.Lambda(lambda x:x[:,PositionTable['vert'][0]:PositionTable['vert'][1]])(news_input)
#     vert_embedding_layer = Embedding(vert_num+1, 200,trainable=True)
#     vert_emb = vert_embedding_layer(vert_input)
#     vert_emb = keras.layers.Reshape((200,))(vert_emb)
#     vert_vec = Dropout(0.2)(vert_emb)
    
#     title_input = keras.layers.Lambda(lambda x:x[:,PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
#     title_emb = word_embedding_layer(title_input)
#     title_emb = Dropout(0.2)(title_emb)
#     title_vecs = Attention(20,20)([title_emb,title_emb,title_emb])

 
    
#     entity_input = keras.layers.Lambda(lambda x:x[:,PositionTable['entity'][0]:PositionTable['entity'][1]])(news_input)
#     entity_embedding_layer = Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],trainable=False)
#     entity_emb = entity_embedding_layer(entity_input)
#     entity_vecs = Attention(20,20)([entity_emb,entity_emb,entity_emb])

#     title_co_emb = Attention(5,40)([title_vecs,entity_vecs,entity_vecs])
#     entity_co_emb = Attention(5,40)([entity_vecs,title_vecs,title_vecs])
    
#     title_vecs = keras.layers.Concatenate(axis=-1)([title_vecs,title_co_emb])
#     title_vecs = Dense(400)(title_vecs)
#     title_vecs = Dropout(0.2)(title_vecs)
#     title_vec = AttentivePooling(config['title_length'],400)(title_vecs)
    
#     entity_vecs = keras.layers.Concatenate(axis=-1)([entity_vecs,entity_co_emb])
#     entity_vecs = Dense(400)(entity_vecs)
#     entity_vecs = Dropout(0.2)(entity_vecs)
#     entity_vec = AttentivePooling(LengthTable['entity'],400)(entity_vecs)
                
#     feature = [title_vec,entity_vec,vert_vec]

#     news_vec = keras.layers.Concatenate(axis=-1)(feature)
#     news_vec = Dense(400)(news_vec)
#     model = Model(news_input,news_vec)
#     return model

# Never used
# def get_news_encoder_co3(config,vert_num,subvert_num,word_num,word_embedding_matrix,entity_embedding_matrix):
#     LengthTable = {'title':config['title_length'],
#                    'vert':1,'subvert':1,
#                    'entity':config['max_entity_num']}
#     input_length = 0
#     PositionTable = {}
#     for v in config['attrs']:
#         PositionTable[v] = (input_length,input_length+LengthTable[v])
#         input_length += LengthTable[v]
#     print(PositionTable)
#     word_embedding_layer = Embedding(word_num+1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],trainable=True)

#     news_input = Input((input_length,),dtype='int32')
    
#     vert_input = keras.layers.Lambda(lambda x:x[:,PositionTable['vert'][0]:PositionTable['vert'][1]])(news_input)
#     vert_embedding_layer = Embedding(vert_num+1, 200,trainable=True)
#     vert_emb = vert_embedding_layer(vert_input)
#     vert_emb = keras.layers.Reshape((200,))(vert_emb)
#     vert_vec = Dropout(0.2)(vert_emb)
    
#     title_input = keras.layers.Lambda(lambda x:x[:,PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
#     title_emb = word_embedding_layer(title_input)
#     title_emb = Dropout(0.2)(title_emb)
#     title_vecs = Attention(20,20)([title_emb,title_emb,title_emb])
#     title_vecs = Dropout(0.2)(title_vecs)
#     title_vec = AttentivePooling(config['title_length'],400)(title_vecs)
 
    
#     entity_input = keras.layers.Lambda(lambda x:x[:,PositionTable['entity'][0]:PositionTable['entity'][1]])(news_input)
#     entity_embedding_layer = Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],trainable=False)
#     entity_emb = entity_embedding_layer(entity_input)
#     entity_vecs = Attention(20,20)([entity_emb,entity_emb,entity_emb])
#     entity_vecs = Dropout(0.2)(entity_vecs)
#     entity_vec = AttentivePooling(LengthTable['entity'],400)(entity_vecs)
    
#     title_query_vec = keras.layers.Reshape((1,400))(title_vec)
#     entity_query_vec = keras.layers.Reshape((1,400))(entity_vec)
#     title_co_vec = Attention(1,100,)([entity_query_vec,title_vecs,title_vecs])
#     entity_co_vec = Attention(1,100,)([title_query_vec,entity_vecs,entity_vecs])
#     title_co_vec = keras.layers.Reshape((100,))(title_co_vec)
#     entity_co_vec = keras.layers.Reshape((100,))(entity_co_vec)
    
#     title_vec = keras.layers.Concatenate(axis=-1)([title_vec,title_co_vec])
#     entity_vec = keras.layers.Concatenate(axis=-1)([entity_vec,entity_co_vec])
#     title_vec = Dense(400)(title_vec)
#     entity_vec = Dense(400)(entity_vec)
#     feature = [title_vec,entity_vec,vert_vec]

#     news_vec = keras.layers.Concatenate(axis=-1)(feature)
#     news_vec = Dense(400)(news_vec)
#     model = Model(news_input,news_vec)
#     return model

# timedist module: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
# TODO: can we use this?
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

def create_pe_model(config,model_config,News,word_embedding_matrix,entity_embedding_matrix):
    max_clicked_news = config['max_clicked_news']
        
    if model_config['news_encoder'] == 0:
        news_encoder = get_news_encoder(config,len(News.category_dict),len(News.subcategory_dict),len(News.word_dict),word_embedding_matrix,entity_embedding_matrix)
        bias_news_encoder = get_news_encoder(config,len(News.category_dict),len(News.subcategory_dict),len(News.word_dict),word_embedding_matrix,entity_embedding_matrix) 

    elif model_config['news_encoder'] == 1:
        news_encoder = get_news_encoder_co1(config,len(News.category_dict),len(News.subcategory_dict),len(News.word_dict),word_embedding_matrix,entity_embedding_matrix)
        bias_news_encoder = get_news_encoder_co1(config,len(News.category_dict),len(News.subcategory_dict),len(News.word_dict),word_embedding_matrix,entity_embedding_matrix)


    news_input_length = int(news_encoder.input.shape[1])
    print(news_input_length)
    clicked_input = Input(shape=(max_clicked_news, news_input_length,), dtype='int32')
    clicked_ctr  = Input(shape=(max_clicked_news,),dtype='int32')
    print(clicked_input.shape)
    user_vecs = TimeDistributed(news_encoder)(clicked_input)

    popularity_embedding_layer =  Embedding(200, 400,trainable=True)
    popularity_embedding = popularity_embedding_layer(clicked_ctr)
    
    if model_config['popularity_user_modeling']:
        popularity_embedding_layer =  Embedding(200, 400,trainable=True)
        popularity_embedding = popularity_embedding_layer(clicked_ctr)
        MHSA = Attention(20,20)
        user_vecs = MHSA([user_vecs,user_vecs,user_vecs])
        #user_vec_query = keras.layers.Add()([user_vecs,popularity_embedding])
        user_vec_query = keras.layers.Concatenate(axis=-1)([user_vecs,popularity_embedding])
        user_vec = AttentivePoolingQKY(50,800,400)([user_vec_query,user_vecs])
    else:
        user_vecs = Attention(20,20)([user_vecs,user_vecs,user_vecs])
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news,400)(user_vecs)
    
    candidates = keras.Input((1+config['npratio'],news_input_length,), dtype='int32')
    candidates_ctr = keras.Input((1+config['npratio'],), dtype='float32')
    candidates_rece_emb_index = keras.Input((1+config['npratio'],), dtype='int32')

    if model_config['rece_emb']:
        bias_content_vec = Input(shape=(500,))
        vec1 = keras.layers.Lambda(lambda x:x[:,:400])(bias_content_vec)
        vec2 = keras.layers.Lambda(lambda x:x[:,400:])(bias_content_vec)
        
        vec1 = Dense(256,activation='tanh')(vec1)
        vec1 = Dense(256,activation='tanh')(vec1)
        vec1 = Dense(128,)(vec1)
        bias_content_score = Dense(1,use_bias=False)(vec1)
        
        vec2 = Dense(64,activation='tanh')(vec2)
        vec2 = Dense(64,activation='tanh')(vec2)
        bias_recency_score = Dense(1,use_bias=False)(vec2)
        
        gate = Dense(128,activation='tanh')(bias_content_vec)
        gate = Dense(64,activation='tanh')(gate)
        gate = Dense(1,activation='sigmoid')(gate)
        
        bias_content_score = keras.layers.Lambda(lambda x: (1-x[0])*x[1]+x[0]*x[2] )([gate,bias_content_score,bias_recency_score])
    
        bias_content_scorer = Model(bias_content_vec,bias_content_score)
        
    else:
        bias_content_vec = Input(shape=(400,))
        vec = Dense(256,activation='tanh')(bias_content_vec)
        vec = Dense(256,activation='tanh')(vec)
        vec = Dense(128,)(vec)
        bias_content_score = Dense(1,use_bias=False)(vec)
        bias_content_scorer = Model(bias_content_vec,bias_content_score)
    
    time_embedding_layer = Embedding(1500, 100,trainable=True)
    time_embedding = time_embedding_layer(candidates_rece_emb_index)
    
    candidate_vecs = TimeDistributed(news_encoder)(candidates)
    bias_candidate_vecs = TimeDistributed(bias_news_encoder)(candidates)
    if model_config['rece_emb']:
        bias_candidate_vecs = keras.layers.Concatenate(axis=-1)([bias_candidate_vecs,time_embedding])
    bias_candidate_score = TimeDistributed(bias_content_scorer)(bias_candidate_vecs)
    bias_candidate_score = keras.layers.Reshape((1+config['npratio'],))(bias_candidate_score)
    
    rel_scores = keras.layers.Dot(axes=-1)([user_vec,candidate_vecs])
    
    scaler =  Dense(1,use_bias=False,kernel_initializer=keras.initializers.Constant(value=19))
    ctrs = keras.layers.Reshape((1+config['npratio'],1))(candidates_ctr)
    ctrs = scaler(ctrs)
    bias_ctr_score = keras.layers.Reshape((1+config['npratio'],))(ctrs)
    
    user_activity_input = keras.layers.Input((1,),dtype='int32')
    
    user_vec_input = keras.layers.Input((400,),)
    activity_gate = Dense(128,activation='tanh')(user_vec_input)
    activity_gate = Dense(64,activation='tanh')(user_vec_input)
    activity_gate = Dense(1,activation='sigmoid')(activity_gate)
    activity_gate = keras.layers.Reshape((1,))(activity_gate)
    activity_gater = Model(user_vec_input,activity_gate)
    
    
    user_activtiy = activity_gater(user_vec)
    
    scores = []
    if model_config['rel']:
        if model_config['activity']:
            print(user_activtiy.shape)
            print(rel_scores.shape)
            rel_scores = keras.layers.Lambda(lambda x:2*x[0]*x[1])([rel_scores,user_activtiy])
            print(rel_scores.shape)

        scores.append(rel_scores)
    if model_config['content']:
        if model_config['activity']:
            bias_candidate_score = keras.layers.Lambda(lambda x:2*x[0]*(1-x[1]))([bias_candidate_score,user_activtiy])
        scores.append(bias_candidate_score)
    if model_config['ctr']:
        if model_config['activity']:
            bias_ctr_score = keras.layers.Lambda(lambda x:2*x[0]*(1-x[1]))([bias_ctr_score,user_activtiy])
        scores.append(bias_ctr_score)



    if len(scores)>1:
        scores = keras.layers.Add()(scores)
    else:
        scores = scores[0]
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)

    model = Model([candidates,candidates_ctr,candidates_rece_emb_index,user_activity_input,clicked_input,clicked_ctr], [logits])


    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])

    user_encoder = Model([clicked_input,clicked_ctr],user_vec)
    
    return model,user_encoder,news_encoder,bias_news_encoder,bias_content_scorer,scaler,time_embedding_layer,activity_gater

class Popularity_aware_user_encoder(nn.Module):
    def __init__(self, config, model_config, News, word_embedding_matrix, entity_embedding_matrix):
        self.model_config = model_config
        self.max_clicked_news = config['max_clicked_news']
        
        if model_config['news_encoder'] == 0:
            self.news_encoder = News_encoder(config, len(News.category_dict), len(News.subcategory_dict), len(News.word_dict), word_embedding_matrix, entity_embedding_matrix)
            self.bias_news_encoder = News_encoder(config, len(News.category_dict), len(News.subcategory_dict), len(News.word_dict), word_embedding_matrix, entity_embedding_matrix) 

        elif model_config['news_encoder'] == 1:
            self.news_encoder = News_encoder_co1(config, len(News.category_dict), len(News.subcategory_dict), len(News.word_dict), word_embedding_matrix, entity_embedding_matrix)
            self.bias_news_encoder = News_encoder_co1(config, len(News.category_dict), len(News.subcategory_dict), len(News.word_dict), word_embedding_matrix, entity_embedding_matrix)

        self.news_input_length = int(self.news_encoder.input.shape[1]) # TODO does this work like this in torch?

        self.timeDistributed = TimeDistributed(self.news_encoder)
        self.popularity_embedding_layer = nn.Embedding(200,400) # Embedding(200, 400,trainable=True)
        self.MHSA = Attention(20,20)
        self.attentivePoolQKY = AttentivePoolingQKY(50, 800, 400)
        self.attentivePool = AttentivePooling(self.max_clicked_news, 400)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # [cliked_input, clicked_ctr]
        user_vecs = self.timeDistributed(x[0])
        user_vecs = self.MHSA(user_vecs, user_vecs, user_vecs)
        popularity_embedding = self.popularity_embedding_layer(x[1])

        if (self.model_config['popularity_user_modeling']):
            user_vec_query = torch.cat((user_vecs, popularity_embedding), dim = -1) # keras.layers.Concatenate(axis=-1)([user_vecs,popularity_embedding])
            user_vec = self.attentivePoolQKY(user_vec_query)
        else:
            user_vecs = self.dropout(user_vecs)
            user_vec = self.attentivePool(user_vecs)
        
        return user_vec
        
    


def create_pe_model(config, model_config, News, word_embedding_matrix, entity_embedding_matrix):

    Pop_aware_user_encoder = Popularity_aware_user_encoder(config, model_config, News, word_embedding_matrix, entity_embedding_matrix)
    
    clicked_input = Input(shape=(max_clicked_news, news_input_length,), dtype='int32')
    clicked_ctr  = Input(shape=(max_clicked_news,),dtype='int32')

    candidates = keras.Input((1+config['npratio'],news_input_length,), dtype='int32')
    candidates_ctr = keras.Input((1+config['npratio'],), dtype='float32')
    candidates_rece_emb_index = keras.Input((1+config['npratio'],), dtype='int32')

    if model_config['rece_emb']:
        bias_content_vec = Input(shape=(500,))
        vec1 = keras.layers.Lambda(lambda x:x[:,:400])(bias_content_vec)
        vec2 = keras.layers.Lambda(lambda x:x[:,400:])(bias_content_vec)
        
        vec1 = Dense(256,activation='tanh')(vec1)
        vec1 = Dense(256,activation='tanh')(vec1)
        vec1 = Dense(128,)(vec1)
        bias_content_score = Dense(1,use_bias=False)(vec1)
        
        vec2 = Dense(64,activation='tanh')(vec2)
        vec2 = Dense(64,activation='tanh')(vec2)
        bias_recency_score = Dense(1,use_bias=False)(vec2)
        
        gate = Dense(128,activation='tanh')(bias_content_vec)
        gate = Dense(64,activation='tanh')(gate)
        gate = Dense(1,activation='sigmoid')(gate)
        
        bias_content_score = keras.layers.Lambda(lambda x: (1-x[0])*x[1]+x[0]*x[2] )([gate,bias_content_score,bias_recency_score])
    
        bias_content_scorer = Model(bias_content_vec,bias_content_score)
        
    else:
        bias_content_vec = Input(shape=(400,))
        vec = Dense(256,activation='tanh')(bias_content_vec)
        vec = Dense(256,activation='tanh')(vec)
        vec = Dense(128,)(vec)
        bias_content_score = Dense(1,use_bias=False)(vec)
        bias_content_scorer = Model(bias_content_vec,bias_content_score)
    
    time_embedding_layer = Embedding(1500, 100,trainable=True)
    time_embedding = time_embedding_layer(candidates_rece_emb_index)
    
    candidate_vecs = TimeDistributed(news_encoder)(candidates)
    bias_candidate_vecs = TimeDistributed(bias_news_encoder)(candidates)
    if model_config['rece_emb']:
        bias_candidate_vecs = keras.layers.Concatenate(axis=-1)([bias_candidate_vecs,time_embedding])
    bias_candidate_score = TimeDistributed(bias_content_scorer)(bias_candidate_vecs)
    bias_candidate_score = keras.layers.Reshape((1+config['npratio'],))(bias_candidate_score)
    
    rel_scores = keras.layers.Dot(axes=-1)([user_vec,candidate_vecs])
    
    scaler =  Dense(1,use_bias=False,kernel_initializer=keras.initializers.Constant(value=19))
    ctrs = keras.layers.Reshape((1+config['npratio'],1))(candidates_ctr)
    ctrs = scaler(ctrs)
    bias_ctr_score = keras.layers.Reshape((1+config['npratio'],))(ctrs)
    
    user_activity_input = keras.layers.Input((1,),dtype='int32')
    
    user_vec_input = keras.layers.Input((400,),)
    activity_gate = Dense(128,activation='tanh')(user_vec_input)
    activity_gate = Dense(64,activation='tanh')(user_vec_input)
    activity_gate = Dense(1,activation='sigmoid')(activity_gate)
    activity_gate = keras.layers.Reshape((1,))(activity_gate)
    activity_gater = Model(user_vec_input,activity_gate)
    
    
    user_activtiy = activity_gater(user_vec)
    
    scores = []
    if model_config['rel']:
        if model_config['activity']:
            print(user_activtiy.shape)
            print(rel_scores.shape)
            rel_scores = keras.layers.Lambda(lambda x:2*x[0]*x[1])([rel_scores,user_activtiy])
            print(rel_scores.shape)

        scores.append(rel_scores)
    if model_config['content']:
        if model_config['activity']:
            bias_candidate_score = keras.layers.Lambda(lambda x:2*x[0]*(1-x[1]))([bias_candidate_score,user_activtiy])
        scores.append(bias_candidate_score)
    if model_config['ctr']:
        if model_config['activity']:
            bias_ctr_score = keras.layers.Lambda(lambda x:2*x[0]*(1-x[1]))([bias_ctr_score,user_activtiy])
        scores.append(bias_ctr_score)



    if len(scores)>1:
        scores = keras.layers.Add()(scores)
    else:
        scores = scores[0]
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)

    model = Model([candidates,candidates_ctr,candidates_rece_emb_index,user_activity_input,clicked_input,clicked_ctr], [logits])


    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])

    user_encoder = Model([clicked_input,clicked_ctr],user_vec)
    