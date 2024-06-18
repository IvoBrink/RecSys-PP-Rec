# from keras.layers import Embedding, concatenate
# from keras.layers import Dense, Input, Flatten, average,Lambda

# from keras.layers import *
# from keras.models import Model, load_model
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# from keras import backend as K
# import keras.layers as layers
# from keras.engine.topology import Layer, InputSpec
# from keras import initializers #keras2
# from keras.utils import plot_model
# from keras.optimizers import *
# from keras.utils import Sequence
# import keras

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init


# class Attention(Layer):
#     def __init__(self, nb_head, size_per_head, **kwargs):
#         self.nb_head = nb_head
#         self.size_per_head = size_per_head
#         self.output_dim = nb_head*size_per_head
#         super(Attention, self).__init__(**kwargs)


#     def build(self, input_shape):
#         self.WQ = self.add_weight(name='WQ',
#                                   shape=(input_shape[0][-1], self.output_dim),
#                                   initializer='glorot_uniform',
#                                   trainable=True)
#         self.WK = self.add_weight(name='WK',
#                                   shape=(input_shape[1][-1], self.output_dim),
#                                   initializer='glorot_uniform',
#                                   trainable=True)
#         self.WV = self.add_weight(name='WV',
#                                   shape=(input_shape[2][-1], self.output_dim),
#                                   initializer='glorot_uniform',
#                                   trainable=True)
#         super(Attention, self).build(input_shape)
 
#     def Mask(self, inputs, seq_len, mode='mul'):
#         if seq_len == None:
#             return inputs
#         else:
#             mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
#             mask = 1 - K.cumsum(mask, 1)
#             for _ in range(len(inputs.shape)-2):
#                 mask = K.expand_dims(mask, 2)
#             if mode == 'mul':
#                 return inputs * mask
#             if mode == 'add':
#                 return inputs - (1 - mask) * 1e12
 
#     def call(self, x):
#         if len(x) == 3:
#             Q_seq,K_seq,V_seq = x
#             Q_len,V_len = None,None
#         elif len(x) == 5:
#             Q_seq,K_seq,V_seq,Q_len,V_len = x
#         Q_seq = K.dot(Q_seq, self.WQ)
#         Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
#         Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
#         K_seq = K.dot(K_seq, self.WK)
#         K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
#         K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
#         V_seq = K.dot(V_seq, self.WV)
#         V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
#         V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

#         A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
#         A = K.permute_dimensions(A, (0,3,2,1))
#         A = self.Mask(A, V_len, 'add')
#         A = K.permute_dimensions(A, (0,3,2,1))
#         A = K.softmax(A)

#         O_seq = K.batch_dot(A, V_seq, axes=[3,2])
#         O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
#         O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
#         O_seq = self.Mask(O_seq, Q_len, 'mul')
#         return O_seq
 
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0][0], input_shape[0][1], self.output_dim)


class Attention(nn.Module):
    def __init__(self, nb_head, size_per_head):
        super().__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        
        # Initialize weights
        self.WQ = nn.Linear(size_per_head, self.output_dim, bias=False)
        self.WK = nn.Linear(size_per_head, self.output_dim, bias=False)
        self.WV = nn.Linear(size_per_head, self.output_dim, bias=False)

    def mask(self, inputs, seq_len, mode='mul'):
        if seq_len is None:
            return inputs
        else:
            mask = torch.ones_like(inputs)
            for i in range(mask.shape[0]):
                mask[i, seq_len[i]:] = 0
            if mode == 'mul':
                return inputs * mask
            elif mode == 'add':
                return inputs - (1 - mask) * 1e12

    def forward(self, Q_seq, K_seq, V_seq, Q_len=None, V_len=None):
        Q_seq = self.WQ(Q_seq)
        Q_seq = Q_seq.view(-1, Q_seq.shape[1], self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        
        K_seq = self.WK(K_seq)
        K_seq = K_seq.view(-1, K_seq.shape[1], self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        
        V_seq = self.WV(V_seq)
        V_seq = V_seq.view(-1, V_seq.shape[1], self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        
        A = torch.matmul(Q_seq, K_seq.transpose(-1, -2)) / (self.size_per_head ** 0.5)
        A = self.mask(A, V_len, 'add')
        A = F.softmax(A, dim=-1)
        
        O_seq = torch.matmul(A, V_seq)
        O_seq = O_seq.permute(0, 2, 1, 3).contiguous().view(-1, O_seq.shape[2], self.output_dim)
        O_seq = self.mask(O_seq, Q_len, 'mul')
        
        return O_seq


# def AttentivePooling(dim1,dim2):
#     vecs_input = Input(shape=(dim1,dim2),dtype='float32') #(50,400)
#     user_vecs =Dropout(0.2)(vecs_input)
#     user_att = Dense(200,activation='tanh')(user_vecs) # (50,200)
#     user_att = Flatten()(Dense(1)(user_att)) # (50,)
#     user_att = Activation('softmax')(user_att)  # (50,)
#     user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])  # (400,)
#     model = Model(vecs_input,user_vec)
#     return model


class AttentivePooling(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        
        print(dim1, dim2)
        self.dropout = nn.Dropout(0.2)
        self.dense_tanh = nn.Linear(dim1, 200)
        self.dense_flat = nn.Linear(200, 1)
        
    def forward(self, x):
        x = self.dropout(x)
        att = torch.tanh(self.dense_tanh(x))
        print(att.shape, "att")
        att = self.dense_flat(att).squeeze(-1)
        att = F.softmax(att, dim=-1)
        att = att.unsqueeze(-1)
        print(att.shape, "att")
        output = torch.sum(x * att, dim=1)
        return output

# def AttentivePoolingQKY(dim1,dim2,dim3):
#     vecs_input = Input(shape=(dim1,dim2),dtype='float32')
#     value_input = Input(shape=(dim1,dim3),dtype='float32')
#     user_vecs =Dropout(0.2)(vecs_input)
#     user_att = Dense(200,activation='tanh')(user_vecs)
#     user_att = Flatten()(Dense(1)(user_att))
#     user_att = Activation('softmax')(user_att)
#     user_vec = keras.layers.Dot((1,1))([value_input,user_att])
#     model = Model([vecs_input,value_input],user_vec)
#     return model

class AttentivePoolingQKY(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.dense_tanh = nn.Linear(dim2, 200)
        self.dense_flat = nn.Linear(200, 1)

    def forward(self, x, y):
        x = self.dropout(x)
        att = torch.tanh(self.dense_tanh(x))
        att = self.dense_flat(att).squeeze(-1)
        att = F.softmax(att, dim=-1)
        att = att.unsqueeze(-1)
        output = torch.sum(y * att, dim=1)
        return output


# def AttentivePooling_bias(dim1,dim2,dim3):
#     bias_input = Input(shape=(dim1,dim2),dtype='float32')
#     value_input = Input(shape=(dim1,dim3),dtype='float32')
    
#     bias_vecs =Dropout(0.2)(bias_input)
#     user_att = Dense(200,activation='tanh')(user_vecs)
#     user_att = Flatten()(Dense(1)(user_att))
#     user_att = Activation('softmax')(user_att)
#     user_vec = keras.layers.Dot((1,1))([value_input,user_att])
#     model = Model([vecs_input,value_input],user_vec)
#     return model

class AttentivePoolingBias(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.dense_tanh = nn.Linear(dim2, 200)
        self.dense_flat = nn.Linear(200, 1)

    def forward(self, bias_input, value_input):
        bias_input = self.dropout(bias_input)
        att = torch.tanh(self.dense_tanh(bias_input))
        att = self.dense_flat(att).squeeze(-1)
        att = F.softmax(att, dim=-1)
        att = att.unsqueeze(-1)
        output = torch.sum(value_input * att, dim=1)
        return output
