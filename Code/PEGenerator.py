import torch
from torch.utils.data import Dataset
import numpy as np

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

def fetch_ctr_dim2(News, docids, bucket, flag=1):
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
    ctr = np.array(ctr, dtype='int32')
    # This deals with errors in the data
    if(np.max(ctr) > 199):
        ctr = np.minimum(ctr, 199)
    return ctr

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

        rece = bucket - self.News.news_publish_bucket2[doc_ids]
        
        rece_emb_index = torch.IntTensor(compute_Q_publish(rece))
        user_activity = torch.IntTensor([(clicked_ids > 0).sum(axis=-1)])
        
        label = torch.FloatTensor(self.label[idx])
        input = (news_feature, candidate_ctr, rece_emb_index, user_activity, user_feature, click_ctr)
       
        return input, label

class UserDataset(Dataset):
    def __init__(self, News, Users):
        self.News = News
        self.Users = Users
        self.ImpNum = self.Users.click.shape[0]

    def __len__(self):
        return self.ImpNum 

    def __getitem__(self, idx):
        clicked_ids = self.Users.click[idx]
        user_feature = torch.IntTensor(np.array([self.News.fetch_news(clicked_ids)]))
        click_bucket = self.Users.click_bucket[idx]
        click_ctr = torch.IntTensor(np.array([fetch_ctr_dim3(self.News, clicked_ids, click_bucket, FLAG_CTR)]))

        return user_feature, click_ctr

class NewsDataset(Dataset):
    def __init__(self, News):
        self.News = News
        self.ImpNum = self.News.title.shape[0]

    def __len__(self):
        return self.ImpNum

    def __getitem__(self, idx):
        
        news_feature = torch.IntTensor(self.News.fetch_news(idx))

        return news_feature






