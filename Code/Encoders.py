from NewsContent import *
from UserContent import *
from preprocessing import *
from models import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class Doc_encoder(nn.Module):
    def __init__(self, config, text_length, embedding_layer):
        super().__init__()
        self.news_encoder = config['news_encoder_name']
        self.embedding_layer = embedding_layer
        self.dropout = nn.Dropout(p=0.2)
        self.conv = nn.Conv1d(300, 400, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.attentivePool = AttentivePooling( 400)

    def forward(self, x):
        x = self.embedding_layer(x) 
        x = self.dropout(x)

        if self.news_encoder=='CNN':
            x = torch.permute(x, (0, 2, 1))
            x = self.conv(x)
            x = self.relu(x)

        x = torch.permute(x, (0, 2, 1))
        
        x = self.dropout(x)
        x = self.attentivePool(x)
        return x


        
class Vert_encoder(nn.Module):
    def __init__(self, config, vert_num):
        super().__init__()
        self.embedding_layer = nn.Embedding(vert_num+1, 400) 
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.embedding_layer(x) 
        x = self.dropout(x)
        return x


class News_encoder(nn.Module):
    def __init__(self, config, vert_num, subvert_num, word_num, word_embedding_matrix, entity_embedding_matrix, topic_embedding_matrix):
        super().__init__()
        self.config = config
        self.vert_num = vert_num
        self.subvert_num = subvert_num
        self.word_num = word_num
        self.word_embedding_matrix = word_embedding_matrix
        self.entity_embedding_matrix = entity_embedding_matrix
        self.topic_embedding_matrix = topic_embedding_matrix
        self.LengthTable = {'title':config['title_length'],
                   'body':config['body_length'],
                   'vert':1,'subvert':1,
                   'entity':config['max_entity_num'],
                   'topic': config['max_topics_num']}
        self.input_length = 0
        self.PositionTable = {}
        for v in config['attrs']:
            self.PositionTable[v] = (self.input_length, self.input_length + self.LengthTable[v])
            self.input_length += self.LengthTable[v]
       
        
        self.title_vec = None
        self.body_vec = None
        self.vert_vec = None
        self.subvert_vec = None
        self.entity_vec = None
        self.topic_vec = None

        
        if 'vert' in config['attrs']:
            self.vert_encoder = Vert_encoder(config, vert_num)

        if 'subvert' in config['attrs']:
            self.subvert_encoder = Vert_encoder(config, subvert_num)

        if 'title' in config['attrs'] or 'body' in config['attrs']:
            self.word_embedding_layer = nn.Embedding(self.word_num + 1, word_embedding_matrix.shape[1]) 
            with torch.no_grad():
                self.word_embedding_layer.weight = nn.Parameter(torch.from_numpy(word_embedding_matrix).float())
            if 'title' in config['attrs']:
                self.title_encoder = Doc_encoder(config, config['title_length'], self.word_embedding_layer)
            if 'body' in config['attrs']:
                self.body_encoder = Doc_encoder(config, config['body_length'], self.word_embedding_layer)

        if 'entity' in config['attrs']:
            self.entity_attention = Attention(20,20, 300, 300, 300)
            self.entity_attentive_pool = AttentivePooling(400)
            self.entity_embedding_layer = nn.Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1])
            with torch.no_grad():
                self.entity_embedding_layer.weight = nn.Parameter(torch.from_numpy(entity_embedding_matrix).float())

        if 'topic' in config['attrs']:
            self.topic_attention = Attention(20,20, 300, 300, 300)
            self.topic_attentive_pool = AttentivePooling(400)
            self.topic_embedding_layer = nn.Embedding(topic_embedding_matrix.shape[0], topic_embedding_matrix.shape[1]) 
            with torch.no_grad():
                self.topic_embedding_layer.weight = nn.Parameter(torch.from_numpy(topic_embedding_matrix).float())

        self.final_attentive_pool = AttentivePooling(400)


    def forward(self, x):

        
        if 'title' in self.config['attrs']:
            title_input = (lambda xi: xi[:, self.PositionTable['title'][0] : self.PositionTable['title'][1]]) (x)
            self.title_vec = self.title_encoder(title_input)
        
        if 'body' in self.config['attrs']:
            body_input = (lambda xi: xi[:, self.PositionTable['body'][0] : self.PositionTable['body'][1]]) (x)
            self.body_vec = self.body_encoder(body_input)
        
        if 'vert' in self.config['attrs']:
            vert_input = (lambda xi: xi[:, self.PositionTable['vert'][0] : self.PositionTable['vert'][1]]) (x)
            self.vert_vec = self.vert_encoder(vert_input)
        
        if 'subvert' in self.config['attrs']:
            subvert_input = (lambda xi: xi[:, self.PositionTable['subvert'][0] : self.PositionTable['subvert'][1]]) (x)
            self.subvert_vec = self.subvert_encoder(subvert_input)
        
        if 'entity' in self.config['attrs']:
            entity_input = (lambda xi: xi[:, self.PositionTable['entity'][0] : self.PositionTable['entity'][1]]) (x)
            entity_emb = self.entity_embedding_layer(entity_input)
            entity_vecs = self.entity_attention(entity_emb,entity_emb,entity_emb)
            self.entity_vec = self.entity_attentive_pool(entity_vecs)

        if 'topic' in self.config['attrs']:
            topic_input = (lambda xi: xi[:, self.PositionTable['topic'][0] : self.PositionTable['topic'][1]]) (x)
            topic_emb = self.topic_embedding_layer(topic_input)
            topic_vecs = self.topic_attention(topic_emb,topic_emb,topic_emb)
            self.topic_vec = self.topic_attentive_pool(topic_vecs)

        vec_Table = {'title':self.title_vec,'body':self.body_vec,'vert':self.vert_vec, 'entity': self.entity_vec, 'subvert': self.subvert_vec, 'topic': self.topic_vec}
        feature = []
        for attr in self.config['attrs']:
            feature.append(vec_Table[attr])

        if len(feature)==1:
            news_vec = feature[0]
        else:
            for i in range(len(feature)):
                feature[i] = torch.reshape(feature[i], (-1, 1, 400)) 
            
            news_vecs = torch.cat(feature, dim = 1)   
            news_vec = self.final_attentive_pool(news_vecs)

        return news_vec


class News_encoder_co1(nn.Module):
    def __init__(self, config, vert_num, subvert_num, word_num, word_embedding_matrix, entity_embedding_matrix, image_emb_matrix):
        super().__init__()
        self.config = config
        self.vert_num = vert_num
        self.subvert_num = subvert_num
        self.word_num = word_num
        self.word_embedding_matrix = word_embedding_matrix
        self.entity_embedding_matrix = entity_embedding_matrix
        self.image_emb_mat = image_emb_matrix
        self.LengthTable = {'title':config['title_length'],
                   'vert':1,'subvert':1,
                   'entity':config['max_entity_num'],
                   'topic':config['max_topics_num'],
                   'image' : config['image_emb_len']}
        self.input_length = 0
        self.PositionTable = {}
        for v in config['attrs']:
            self.PositionTable[v] = (self.input_length, self.input_length + self.LengthTable[v])
            self.input_length += self.LengthTable[v]
        
        self.word_embedding_layer = nn.Embedding(self.word_num + 1, word_embedding_matrix.shape[1]) 
        self.entity_embedding_layer = nn.Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1])  
        self.vert_embedding_layer = nn.Embedding(self.vert_num + 1, 200) 
        if ('image' in self.config['attrs']):
            self.image_embedding_layer = nn.Embedding(image_emb_matrix.shape[0], image_emb_matrix.shape[1])
            self.image_embedding_layer.weight.requires_grad = False
        
        
        with torch.no_grad():
            self.word_embedding_layer.weight = nn.Parameter(torch.from_numpy(word_embedding_matrix).float())
            self.entity_embedding_layer.weight = nn.Parameter(torch.from_numpy(entity_embedding_matrix).float())
            if ('image' in self.config['attrs']):
                self.image_embedding_layer.weight = nn.Parameter(torch.from_numpy(image_emb_matrix).float())

        self.attention = Attention(20,20,300, 300, 300)
        self.attention2 = Attention(5,40, 300, 300, 300)
        self.attentive_pool = AttentivePooling(400)
        self.attentive_pool2 = AttentivePooling(400)
        self.dropout = nn.Dropout(p = 0.2)
        self.fc1 = nn.LazyLinear(400)
        self.fc2 = nn.LazyLinear(400)
        self.fc3 = nn.LazyLinear(400)
        
        self.title_vec = None
        self.body_vec = None
        self.vert_vec = None
        self.subvert_vec = None
        self.entity_vec = None

    def forward(self, x):

        vert_input = (lambda xi: xi[:, self.PositionTable['vert'][0] : self.PositionTable['vert'][1]]) (x)
        vert_emb = self.vert_embedding_layer(vert_input)
        vert_emb = vert_emb.view(-1, 200)
        vert_vec = self.dropout(vert_emb)

        title_input = (lambda xi: xi[:, self.PositionTable['title'][0] : self.PositionTable['title'][1]]) (x)
        title_emb = self.word_embedding_layer(title_input)
        title_emb = self.dropout(title_emb)

        entity_input = (lambda xi: xi[:, self.PositionTable['entity'][0] : self.PositionTable['entity'][1]])  (x)
        entity_emb = self.entity_embedding_layer(entity_input)

        if ('image' in self.config['attrs']):
            image_imp = (lambda xi: xi[:, self.PositionTable['image'][0] : self.PositionTable['image'][1]])  (x)
            image_emb = self.image_embedding_layer(image_imp)
            image_emb = image_emb.view(-1, 1024)
            image_vec = self.dropout(image_emb)

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


        if ('image' in self.config['attrs']):
            feature = [title_vec, entity_vec, vert_vec, image_vec]
        else:
            feature = [title_vec, entity_vec, vert_vec]

        news_vec = torch.cat(feature, dim = -1)
        news_vec = self.fc3(news_vec)
        return news_vec

# TimeDistributed module taken from: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
# This is used to take an input of the shape [batch, group, features] to [batch*group, features], apply a module, and then
# revert them back to the shape [batch, group, new_features]
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) < 2:
            return self.module(x)
        
        x_reshape = x.contiguous().view(-1, x.size(-1))  

        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  
        else:
            y = y.view(-1, x.size(1), y.size(-1))  

        return y

    def to(self, device):
        self.module.to(device)
        super().to(device)

class Popularity_aware_user_encoder(nn.Module):
    def __init__(self, config, model_config, news_encoder):
        super().__init__()
        self.model_config = model_config
        self.max_clicked_news = config['max_clicked_news']
        self.news_encoder = news_encoder
        
        self.timeDistributed = TimeDistributed(self.news_encoder)
        self.popularity_embedding_layer = nn.Embedding(200,400) 
        self.MHSA = Attention(20,20, 400, 400, 400)
        self.attentivePool = AttentivePooling(400)
        self.dropout = nn.Dropout(0.2)
        if (self.model_config['popularity_user_modeling']):
            self.attentivePoolQKY = AttentivePoolingQKY(800)

    def forward(self, x):
        user_vecs = self.timeDistributed(x[0])
        user_vecs = self.MHSA(user_vecs, user_vecs, user_vecs)
        popularity_embedding = self.popularity_embedding_layer(x[1])

        if (self.model_config['popularity_user_modeling']):
            user_vec_query = torch.cat((user_vecs, popularity_embedding), dim = -1) 
            user_vec = self.attentivePoolQKY(user_vec_query, user_vecs)
        else:
            user_vecs = self.dropout(user_vecs)
            user_vec = self.attentivePool(user_vecs)
        
        return user_vec
        
class Bias_content_scorer(nn.Module):
    def __init__(self, config, model_config):
        super().__init__()
        self.model_config = model_config
        self.max_clicked_news = config['max_clicked_news']
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1_1 = nn.Linear(400, 256)
        self.fc1_2 = nn.Linear(256, 256)
        self.fc1_3 = nn.Linear(256, 128) 
        self.fc1_4 = nn.Linear(128, 1, bias=False)

        if (self.model_config['rece_emb']):

            self.fc2_1 = nn.Linear(100, 64)
            self.fc2_2 = nn.Linear(64, 64)
            self.fc2_3 = nn.Linear(64, 1, bias=False)

            self.fc3_1 = nn.Linear(500, 128)
            self.fc3_2 = nn.Linear(128, 64)
            self.fc3_3 = nn.Linear(64, 1)
        
    def forward(self, x):

        if self.model_config['rece_emb']:
            vec2 = (lambda xi:xi[:, 400:]) (x) 

            gate = self.tanh(self.fc3_1(x))
            gate = self.tanh(self.fc3_2(gate))
            gate = self.sigmoid(self.fc3_3(gate)) 

            x = (lambda xi:xi[:, :400]) (x) 
            
            vec2 = self.tanh(self.fc2_1(vec2))
            vec2 = self.tanh(self.fc2_2(vec2))
            bias_recency_score = self.fc2_3(vec2)

        vec = self.tanh(self.fc1_1(x))
        vec = self.tanh(self.fc1_2(vec))
        vec = self.fc1_3(vec)
        bias_content_score = self.fc1_4(vec)

        if self.model_config['rece_emb']:
            bias_content_score = (lambda x: (1-x[0]) * x[1] + x[0] * x[2])([gate, bias_content_score, bias_recency_score])

        return bias_content_score
    
class Activity_gater(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(400,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        activity_gate = self.tanh(self.fc1(x)) 
        activity_gate = self.tanh(self.fc2(activity_gate)) 
        activity_gate = self.sigmoid(self.fc3(activity_gate)) 
        activity_gate = activity_gate.view(-1,1) 
        return activity_gate 
    
class Multiply(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = nn.Parameter(torch.as_tensor([scaler], dtype=torch.float32))

    def forward(self, x):
        x = torch.multiply(x, self.scaler)
        return x

    
class PE_model(nn.Module):
    def __init__(self, config, model_config, News, word_embedding_matrix, entity_embedding_matrix, image_emb_matrix, topic_emb_matrix, device):
        super().__init__()
        self.device = device
        self.config = config
        self.model_config = model_config
        self.News = News
        self.word_emb_mat = word_embedding_matrix
        self.entity_emb_mat = entity_embedding_matrix
        self.image_emb_mat = image_emb_matrix
        self.topic_emb_mat = topic_emb_matrix

        self.pop_aware_user_encoder = None
        self.news_encoder = None
        self.bias_news_encoder = None
        self.bias_scorer = None
        self.scaler = None
        self.time_embedding_layer = None
        self.activity_gater = None

        if model_config['rel']:
            if model_config['news_encoder'] == 0:
                self.news_encoder = News_encoder(self.config, len(self.News.category_dict), len(self.News.subcategory_dict), len(self.News.word_dict), self.word_emb_mat, self.entity_emb_mat, self.topic_emb_mat)
            elif model_config['news_encoder'] == 1:
                self.news_encoder = News_encoder_co1(self.config, len(self.News.category_dict), len(self.News.subcategory_dict), len(self.News.word_dict), self.word_emb_mat, self.entity_emb_mat, self.image_emb_mat)
            self.time_distributed1 = TimeDistributed(self.news_encoder)
            self.pop_aware_user_encoder = Popularity_aware_user_encoder(self.config, self.model_config, self.news_encoder)

        if model_config['content']:
            if model_config['news_encoder'] == 0:
                self.bias_news_encoder = News_encoder(self.config, len(self.News.category_dict), len(self.News.subcategory_dict), len(self.News.word_dict), self.word_emb_mat, self.entity_emb_mat, self.topic_emb_mat) 
            elif model_config['news_encoder'] == 1:
                self.bias_news_encoder = News_encoder_co1(self.config, len(self.News.category_dict), len(self.News.subcategory_dict), len(self.News.word_dict), self.word_emb_mat, self.entity_emb_mat, self.image_emb_mat)
            if model_config['rece_emb']:
                self.time_embedding_layer = nn.Embedding(1500, 100)
            self.bias_scorer = Bias_content_scorer(self.config, self.model_config)
            self.time_distributed2 = TimeDistributed(self.bias_news_encoder)
            self.time_distributed3 = TimeDistributed(self.bias_scorer)

        if model_config['ctr']:
            self.scaler = Multiply(19)
        
        if model_config['activity']:
            self.activity_gater = Activity_gater()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # attributes of x:
        # 0 candidates
        # 1 candidates ctr
        # 2 candidates rece
        # 3 user activity
        # 4 clicked input
        # 5 clicked ctr        

        # calculating scores
        if self.model_config['rel']:
            candidate_vecs = self.time_distributed1(x[0]).to(self.device) 
            user_vec = self.pop_aware_user_encoder([x[4], x[5]])
            rel_scores = torch.zeros(candidate_vecs.shape[0:2]).to(self.device)
            for i in range(x[0].shape[0]):
                rel_scores[i] = torch.sum(torch.mul(candidate_vecs[i],user_vec[i]), dim = 1).to(self.device)

        if self.model_config['content']:
            bias_candidate_vecs = self.time_distributed2(x[0]).to(self.device)
            if self.model_config['rece_emb']:
                time_embedding = self.time_embedding_layer(x[2]).to(self.device) 
                bias_candidate_vecs = torch.cat((bias_candidate_vecs, time_embedding), dim=-1) 
            bias_candidate_score = self.time_distributed3(bias_candidate_vecs)
            bias_candidate_score = bias_candidate_score.view( -1, 1 + self.config['npratio']).to(self.device)

        if self.model_config['ctr']:
            ctrs = x[1].view(-1, 1)
            ctrs = self.scaler(ctrs)
            bias_ctr_score = ctrs.view(-1, 1 + self.config['npratio']).to(self.device)

        if( self.model_config['activity'] ):
            user_activity = self.activity_gater(user_vec).to(self.device)

        # adding up scores
        scores = []
        if self.model_config['rel']: # user history x news  
            if self.model_config['activity']:
                rel_scores = (lambda x:x[0]*x[1]) ([rel_scores, user_activity])
            scores.append(rel_scores)
        if self.model_config['content']: # news attributes + recency
            if self.model_config['activity']:
                bias_candidate_score = (lambda x:x[0]*(1-x[1]))([bias_candidate_score, user_activity])
            scores.append(bias_candidate_score)
        if self.model_config['ctr']: # ctr 
            if self.model_config['activity']:
                bias_ctr_score = (lambda x:x[0]*(1-x[1]))([bias_ctr_score, user_activity])
            scores.append(bias_ctr_score)

        if len(scores)>1:
            scores = torch.stack(scores, dim = 0 )
            scores = torch.sum(scores, dim = 0)
        else:
            scores = scores[0]
        
        logits = self.softmax(scores)
        return logits


def create_pe_model(config, model_config, News, word_embedding_matrix, entity_embedding_matrix, image_emb_matrix, topic_emb_matrix, device):

    if not model_config['rel']:
        model_config['activity'] = False
        model_config['popularity_user_modeling'] = False
    if not model_config['content']:
        model_config['rece'] = False

    model = PE_model(config, model_config, News, word_embedding_matrix, entity_embedding_matrix, image_emb_matrix, topic_emb_matrix, device)

    user_encoder = model.pop_aware_user_encoder
    news_encoder = model.news_encoder
    bias_news_encoder = model.bias_news_encoder
    bias_content_scorer = model.bias_scorer
    scaler = model.scaler
    time_embedding_layer = model.time_embedding_layer
    activity_gater = model.activity_gater

    return model,user_encoder,news_encoder,bias_news_encoder,bias_content_scorer,scaler,time_embedding_layer,activity_gater

    