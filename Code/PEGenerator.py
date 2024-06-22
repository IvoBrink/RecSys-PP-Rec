# from keras.utils import Sequence
# import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys


# FLAG_CTR = 1

# def compute_Q_publish(publish_time):
#     arg = publish_time<0
#     publish_time[arg] = 0
#     return publish_time//2

# def fetch_ctr_dim1(News,docids,bucket,flag = 1):
#     doc_imp = News.news_stat_imp[docids]
#     doc_click = News.news_stat_click[docids]
    
#     if flag == 1:
#         ctr = doc_click[:,bucket-1]/(doc_imp[:,bucket-1]+0.01)
#     return ctr


FLAG_CTR = 1

def compute_Q_publish(publish_time):
    publish_time[publish_time < 0] = 0
    return publish_time // 2

def fetch_ctr_dim1(News, docids, bucket, flag=1):
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]
    
    if flag == 1:
        ctr = doc_click[:, bucket-1] / (doc_imp[:, bucket-1] + 0.01)
    return ctr

# def fetch_ctr_dim2(News,docids,bucket,flag = 1):
#     batch_size, doc_num = docids.shape
#     ctr = np.zeros(docids.shape)
#     doc_imp = News.news_stat_imp[docids]
#     doc_click = News.news_stat_click[docids]
#     for i in range(batch_size):
#         if flag == 1:
#             ctr[i,:] = doc_click[i,:,bucket[i]-1]/(doc_imp[i,:,bucket[i]-1]+0.01)
    
#     return ctr

# def fetch_ctr_dim3(News,docids,bucket,flag = 1):
#     batch_size, doc_num = docids.shape
#     doc_imp = News.news_stat_imp[docids]
#     doc_click = News.news_stat_click[docids]
#     ctr = np.zeros(docids.shape)
#     for i in range(batch_size):
#         for j in range(doc_num):
#             b = bucket[i,j]-1
#             if b<0:
#                 b = 0
#             ctr[i,j] = doc_click[i,j,b]/(doc_imp[i,j,b]+0.01)
#     ctr = ctr*200
#     ct = np.ceil(ctr)
#     ctr = np.array(ctr,dtype='int32')
#     return ctr


def fetch_ctr_dim2(News, docids, bucket, flag=1):
    # doc_num = docids.shape
    ctr = np.zeros(docids.shape)
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]

    if flag == 1:
        ctr = doc_click[ :, bucket - 1] / (doc_imp[ :, bucket - 1] + 0.01)
    
    return ctr

def fetch_ctr_dim3(News, docids, bucket, flag=1):
    doc_num = docids.shape
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]
    ctr = np.zeros(docids.shape)
    for j in range(doc_num[0]):
        b = bucket[j] - 1
        if b < 0:
            b = 0
        ctr[ j] = doc_click[j, b] / (doc_imp[j, b] + 0.01)
    
    ctr = ctr * 200
    # ct = np.ceil(ctr)
    ctr = np.array(ctr, dtype='int32')
    if(np.max(ctr) > 199):
        ctr = np.minimum(ctr, 199)
    # print(np.max(ctr))
    # print(np.max(ctr))
    return ctr


# class TrainGenerator(Sequence):
#     def __init__(self, News,Users, news_id, userids, buckets,label, batch_size):
        
#         self.News = News
#         self.Users = Users

#         self.userids = userids        
#         self.doc_id = news_id
#         self.buckets = buckets
#         self.label = label
        
        
#         self.batch_size = batch_size
#         self.ImpNum = self.label.shape[0]

#     def __len__(self):
#         return int(np.ceil(self.ImpNum / float(self.batch_size)))

#     def __getitem__(self, idx):
#         start = idx*self.batch_size
#         ed = (idx+1)*self.batch_size
#         if ed> self.ImpNum:
#             ed = self.ImpNum
        
#         doc_ids = self.doc_id[start:ed]
#         news_feature = self.News.fetch_news(doc_ids)
        
#         userids = self.userids[start:ed]
#         clicked_ids = self.Users.click[userids]
#         user_feature = self.News.fetch_news(clicked_ids)
        
#         bucket = self.buckets[start:ed]
#         candidate_ctr = fetch_ctr_dim2(self.News,doc_ids,bucket,FLAG_CTR)
        
#         click_bucket = self.Users.click_bucket[userids]
#         click_ctr = fetch_ctr_dim3(self.News,clicked_ids,click_bucket,FLAG_CTR)

#         bucket = bucket.reshape((bucket.shape[0],1))
#         rece = bucket - self.News.news_publish_bucket2[doc_ids]
        
#         rece_emb_index = compute_Q_publish(rece)
#         user_activity = (clicked_ids>0).sum(axis=-1)
        
#         label = self.label[start:ed]
                
#         return ([news_feature,candidate_ctr,rece_emb_index,user_activity,user_feature,click_ctr,],[label])
# class EbnerdTrainData(Dataset):
#     def __init__(self, News, User, session, user_id, buckets, labels):
#         self.News = News
#         self.User = User
#         self.session = session
#         self.buckets = buckets
#         self.user_id = user_id
#         self.labels = labels
#         self.length = len(self.session)
#         # ?
#         # self.batch_size = batch_size
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, idx):
#         candidates = torch.IntTensor(self.session[idx])
#         candidates_data = torch.LongTensor(self.News.fetch_news(candidates))
#         candidates_ctr = torch.FloatTensor([self.News.get_ctr(x, self.buckets[idx]) for x in candidates])
#         candidates_rece_emb_index = torch.IntTensor([self.News.news_publish_bucket2[x] for x in candidates])
#         user_activity_input = 1
#         clicked_input = torch.IntTensor(self.User.click[self.user_id[idx]])
#         clicked_input_data = torch.LongTensor(self.News.fetch_news(clicked_input))
#         # print(clicked_input)
#         clicked_ctr = torch.FloatTensor([self.News.get_ctr(self.User.click[self.user_id[idx]][i], self.User.click_bucket[self.user_id[idx]][i]) for i in range(len(clicked_input))])
#         # print(clicked_ctr)
#         input = [candidates_data,candidates_ctr,candidates_rece_emb_index,user_activity_input,clicked_input_data,clicked_ctr]
#         # for x in range(len(input)):
#         #     input[x] = torch.FloatTensor(input[x])
#         labels = torch.FloatTensor(self.labels[idx])
#         return input, labels


class TrainDataset(Dataset):
    def __init__(self, News, Users, news_id, userids, buckets, label):
        self.News = News
        self.Users = Users
        self.news_id = news_id
        self.userids = userids
        self.buckets = buckets
        self.label = label
        self.ImpNum = label.shape[0]

    def __len__(self):
        return int(self.ImpNum)

    def __getitem__(self, idx):
        doc_ids = self.news_id[idx]
        news_feature = torch.IntTensor(self.News.fetch_news(doc_ids))
        
        userids = self.userids[idx]
        clicked_ids = self.Users.click[userids]
        user_feature = torch.IntTensor(self.News.fetch_news(clicked_ids))
        
        bucket = self.buckets[idx]
        candidate_ctr = torch.FloatTensor(fetch_ctr_dim2(self.News, doc_ids, bucket, FLAG_CTR))
        
        click_bucket = self.Users.click_bucket[userids]
        click_ctr = torch.IntTensor(fetch_ctr_dim3(self.News, clicked_ids, click_bucket, FLAG_CTR))

        # bucket = bucket.reshape((bucket.shape[0], 1))
        # print(bucket)
        # print(self.News.news_publish_bucket2[doc_ids])
        rece = bucket - self.News.news_publish_bucket2[doc_ids]
        
        rece_emb_index = torch.IntTensor(compute_Q_publish(rece))
        user_activity = torch.IntTensor([(clicked_ids > 0).sum(axis=-1)])
        
        label = torch.FloatTensor(self.label[idx])
        input = (news_feature, candidate_ctr, rece_emb_index, user_activity, user_feature, click_ctr)
        # for i in range(len(input)):
        #     print(input[i].shape)
        
        return input, label

# class UserGenerator(Sequence):
#     def __init__(self,News, Users,batch_size):
        
#         self.News = News
#         self.Users = Users

#         self.batch_size = batch_size
#         self.ImpNum = self.Users.click.shape[0]
        
#     def __len__(self):
#         return int(np.ceil(self.ImpNum / float(self.batch_size)))
            
    
#     def __getitem__(self, idx):
#         start = idx*self.batch_size
#         ed = (idx+1)*self.batch_size
#         if ed> self.ImpNum:
#             ed = self.ImpNum

#         clicked_ids = self.Users.click[start:ed]
#         user_feature = self.News.fetch_news(clicked_ids)
#         click_bucket = self.Users.click_bucket[start:ed]
#         click_ctr = fetch_ctr_dim3(self.News,clicked_ids,click_bucket,FLAG_CTR)


#         return [user_feature,click_ctr]




class UserDataset(Dataset):
    def __init__(self, News, Users):
        self.News = News
        self.Users = Users
        self.ImpNum = self.Users.click.shape[0]

    def __len__(self):
        return self.ImpNum 

    def __getitem__(self, idx):
        

        clicked_ids = self.Users.click[idx]
        # print(clicked_ids)
        user_feature = torch.IntTensor([self.News.fetch_news(clicked_ids)])
        click_bucket = self.Users.click_bucket[idx]
        click_ctr = torch.IntTensor([fetch_ctr_dim3(self.News, clicked_ids, click_bucket, FLAG_CTR)])

        return user_feature, click_ctr

# class NewsGenerator(Sequence):
#     def __init__(self,News,batch_size):
#         self.News = News

#         self.batch_size = batch_size        
#         self.ImpNum = self.News.title.shape[0]
        
#     def __len__(self):
#         return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
#     def __getitem__(self, idx):
#         start = idx*self.batch_size
#         ed = (idx+1)*self.batch_size
#         if ed> self.ImpNum:
#             ed = self.ImpNum
#         docids = np.array([i for i in range(start,ed)])
        
#         news_feature = self.News.fetch_news(docids)
        
#         return news_feature


class NewsDataset(Dataset):
    def __init__(self, News):
        self.News = News
        self.ImpNum = self.News.title.shape[0]

    def __len__(self):
        return self.ImpNum

    def __getitem__(self, idx):
        
        news_feature = torch.IntTensor(self.News.fetch_news(idx))

        return news_feature






