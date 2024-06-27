import numpy as np
import torch
from datetime import datetime
import time
import re
from PEGenerator import *
from tqdm import trange

FLAG_CTR = 1

def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def trans2tsp(timestr):
    return int(time.mktime(datetime.strptime(timestr, '%m/%d/%Y %I:%M:%S %p').timetuple()))

anchor = trans2tsp('05/17/2023 11:59:59 PM')
def parse_time_bucket(date):
    tsp = trans2tsp(date)
    tsp = tsp - anchor
    tsp = tsp//(1200 )
    return tsp

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def my_auc(label,score):
    false_score = score[label==0]
    positive_score = score[label==1]
    num_positive = (label==1).sum()
    num_negative = (label==0).sum()
    if num_positive==0:
        return 0.75
    if num_negative==0:
        return 1
    positive_score = positive_score.reshape((num_positive,1))
    positive_score = np.repeat(positive_score,num_negative,axis=1)
    false_score = false_score.reshape((1,num_negative))
    false_score = np.repeat(false_score,num_positive,axis=0)
    return 1-((positive_score<false_score).mean()+0.5*(positive_score==false_score).mean())

def ILAD(vecs):
    score = torch.matmul(vecs,vecs.T)
    score = (score+1)/2
    score = score.mean()-1/score.shape[0]
    score = float(score)
    return score

def ILMD(vecs):
    score = torch.matmul(vecs,vecs.T)
    score = (score+1)/2
    score = score.min()
    score = float(score)
    return score

def eval_model(model_config, News, user_encoder, impressions, user_data, user_ids, news_encoder, bias_news_encoder, activity_gater, 
               time_embedding_layer, bias_content_scorer, scaler, colds, topKs, Users, device):
    
    if scaler is None:
        eval_scaler = 1
    else:
        eval_scaler = scaler.scaler[0]

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]

    coldAUC = []
    coldMRR = []
    coldnDCG5 = []
    coldnDCG10 =[]
    cold_index = {}

    topics = []

    ILADs = []
    ILMDs = []

    for i in range(topKs):
        topics.append([])
        ILADs.append([])
        ILMDs.append([])

    for i in range(len(colds)):
        coldAUC.append([])
        coldMRR.append([])
        coldnDCG5.append([])
        coldnDCG10.append([])
        cold_index[colds[i]] = i

    for i in trange(len(impressions)):
        docids = impressions[i]['docs']
        docids = np.array(docids)
        bucket = impressions[i]['tsp']
        
        publish_time = News.news_publish_bucket2[docids]

        user_coldness = 0

        if model_config['rel']:
            user_data_on_ids = user_data.__getitem__(user_ids[i])
            user_coldness = int(torch.count_nonzero(torch.sum(user_data_on_ids[0], dim=2)[0]))
            user_data_on_ids = (user_data_on_ids[0].to(device), user_data_on_ids[1].to(device))
            uv = user_encoder(user_data_on_ids)
            nv = news_encoder(torch.IntTensor(np.array(News.fetch_news(docids))).to(device))
            rel_score = torch.matmul(nv, uv[0])
            predicted_activity_gate = activity_gater(uv)
            predicted_activity_gate = predicted_activity_gate[:,0]
        else:
            rel_score = 0
        
        if model_config['content'] and model_config['rece_emb']:
            bias_vecs = bias_news_encoder(torch.IntTensor(News.fetch_news(docids)).to(device))
            publish_time = bucket - publish_time
            arg = publish_time < 0
            publish_time[arg] = 0
            publish_bucket = compute_Q_publish(publish_time)
            time_emb = time_embedding_layer.weight[publish_bucket].to(device)
            bias_vecs = torch.cat([bias_vecs,time_emb], axis=-1).to(device)
            bias_score = bias_content_scorer(bias_vecs)
            bias_score = bias_score[:,0]
        elif model_config['content'] and not model_config['rece_emb']:
            bias_vecs = bias_news_encoder(torch.IntTensor(News.fetch_news(docids)).to(device))
            bias_score = bias_content_scorer(bias_vecs)
            bias_score = bias_score[:,0]
        else:
            bias_score = 0


        if model_config['activity']:
            gate = predicted_activity_gate
        else:
            gate = 0.5
        
        if model_config['ctr']:
            ctr = torch.FloatTensor(fetch_ctr_dim1(News,docids,bucket,FLAG_CTR)).to(device)
        else:
            ctr = 0

        score = gate*rel_score + (torch.tensor(1)-gate)*(ctr*eval_scaler + bias_score)
        score = score.cpu().detach().numpy()

        labels = impressions[i]['labels']
        labels = np.array(labels)
        
        # Standard metrics
        auc = my_auc(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

        # Cold user calc
        if user_coldness in colds:
            idx = cold_index[user_coldness]
            coldAUC[idx].append(auc)
            coldMRR[idx].append(mrr)
            coldnDCG5[idx].append(ndcg5)
            coldnDCG10[idx].append(ndcg10)
        
        
        for TOP_DIVERSITY_NUM in range(1, topKs+1):
            # Diversity calc
            top_args = score.argsort()[-TOP_DIVERSITY_NUM:]
            top_docids = docids[top_args]
            verts = News.vert[top_docids].tolist()
            mask = labels[top_args]
            verts = verts * mask
            verts = np.array(verts,dtype='int32')
            
            uc = Users.click[i]
            uverts = News.vert[uc].tolist()
            uverts = set(uverts) -{0}
            
            s = 0
            for v in verts:
                if v == 0:
                    continue
                if not v in uverts:
                    s += 1
            s /= (mask.sum()+0.01)
            topics[TOP_DIVERSITY_NUM-1].append(s)

            # Density calc
            if model_config['rel']:
                nv2 = nv/torch.sqrt(torch.sum(torch.square(nv), dim = -1)).reshape((nv.shape[0],1))
                nv2 = nv2[top_args]
                ilad = ILAD(nv2)
                ilmd = ILMD(nv2)
                ILADs[TOP_DIVERSITY_NUM-1].append(ilad)
                ILMDs[TOP_DIVERSITY_NUM-1].append(ilmd)

    normal_metrics = [AUC, MRR, nDCG5, nDCG10]
    cold_metrics = [coldAUC, coldMRR, coldnDCG5, coldnDCG10]        

    for i in range(len(normal_metrics)):
        normal_metrics[i] = np.array(normal_metrics[i]).mean()

        for j in range(len(colds)):
            cold_metrics[i][j] = np.array(cold_metrics[i][j]).mean()
    
    for i in range(len(topics)):
        topics[i] = np.array(topics[i]).mean()
        ILADs[i] = np.array(ILADs[i]).mean()
        ILMDs[i] = np.array(ILMDs[i]).mean()

    return normal_metrics, cold_metrics, topics, ILADs, ILMDs