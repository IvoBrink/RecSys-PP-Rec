from utils import *
import numpy as np
import random
import os

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

class UserContent():
    def __init__(self,news_index,config,filename,flag):
        self.flag = flag
        self.news_index = news_index
        self.config = config
        self.filename = filename
        self.load_session()
        self.parse_user()

    def load_session(self,):
        config = self.config
        path = config['data_root_path']
        with open(os.path.join(path,self.filename)) as f:
            lines=f.readlines()
     
        session=[]
        print(len(lines))
        for i in range(len(lines)):
            _, clicks, poss, negs, sessionTime = lines[i].strip().split('\t')

            sessionBucket = parse_time_bucket(sessionTime)
            
            click_ids = []
            click_bucket = []
            clicks = clicks.split('), ')
            for i in clicks:
                did, clicktime = i.replace("[","").replace("]","").replace('(',"").replace(")","").replace("'","").split(", ")
        
                bucket = parse_time_bucket(clicktime)
                if (bucket < sessionBucket):
                    click_ids.append(did)
                    click_bucket.append(bucket)

            pos = poss.split(' ')

            neg = negs.split(' ')
            

            session.append([click_ids,click_bucket,sessionBucket,pos,neg])
            

        self.session = session

    def parse_user(self,):
        session = self.session
        config = self.config

        MAX_ALL = config['max_clicked_news']
        news_index = self.news_index

        user_num = len(session)
        self.click = np.zeros((user_num,MAX_ALL),dtype='int32')
        self.click_bucket = np.zeros((user_num,MAX_ALL),dtype='int32')

        for user_id in range(len(session)):
            click_ids,click_bucket,_,_,_ = session[user_id]
            clicks = []
            
            i = 0
            j = len(click_ids)
            while i < j:
                if len(click_ids[i])>10:
                    click_ids.append(click_ids[i].split(' ', 1)[1])
                    click_bucket.append(click_bucket[i])
                    click_ids[i] = click_ids[i].split(' ', 1)[0]
                    j += 1
                i += 1

            for i in range(len(click_ids)):
                clicks.append(news_index[click_ids[i]])

            if len(clicks) >MAX_ALL:
                clicks = clicks[-MAX_ALL:]
                click_bucket = click_bucket[-MAX_ALL:]
            else:
                clicks = [0]*(MAX_ALL-len(click_ids))+clicks
                click_bucket = [1]*(MAX_ALL-len(click_bucket))+click_bucket

                
            self.click[user_id] = np.array(clicks)
            self.click_bucket[user_id] = np.array(click_bucket)
