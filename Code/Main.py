# %%
# %load_ext autoreload
# %autoreload 2

from NewsContent import *
from UserContent import *
from preprocessing import *
from PEGenerator import *
import PEGenerator
from models import *
from utils import *
from Encoders import *

import os
import numpy as np
import json
import random

# %%
data_root_path = "../data/Challenge/"
embedding_path = "../../"
KG_root_path = None
popularity_path = "../data/Challenge/popularity"
config = {'title_length':30,
              'body_length':100,
              'max_clicked_news':50,
              'npratio':1,
              'news_encoder_name':"CNN",
              'user_encoder_name':"Att",
             'attrs':['title','body','vert'],
             'word_filter':0,
             'data_root_path':data_root_path,
             'embedding_path':embedding_path,
             'KG_root_path':KG_root_path,
            'popularity_path':popularity_path,
            'batch_size': 32,
             'max_entity_num':5}
model_config = {
        'news_encoder':0,
        'popularity_user_modeling':True,
        'rel':True,
        'ctr':True,
        'content':True,
        'rece_emb':True,
        'activity':True

    }

# %%
News = NewsContent(config)

TrainUsers = UserContent(News.news_index,config,'train.tsv',2)
ValidUsers = UserContent(News.news_index,config,'val.tsv',1)
TestUsers = UserContent(News.news_index,config,'test.tsv',2)

# %%
train_sess,train_buckets, train_user_id, train_label = get_train_input(TrainUsers.session,News.news_index,config)
test_impressions, test_userids = get_test_input(TestUsers.session,News.news_index)
val_impressions, val_userids = get_test_input(ValidUsers.session,News.news_index)

# %%
title_word_embedding_matrix, have_word = load_matrix(embedding_path,News.word_dict)

# %%
train_loader = TrainDataset(News, TrainUsers, train_sess, train_user_id, train_buckets, train_label, config['batch_size'])
test_user_generator = UserDataset(News,TestUsers, config['batch_size'])
val_user_generator = UserDataset(News,ValidUsers, config['batch_size'])
news_generator = NewsDataset(News, config['batch_size'])



# %%
train_loader.__len__()

# %%
for x, y in train_loader:
    print(type(x[3]), x[3])
    break
    

# %%
title_word_embedding_matrix.shape

# %%
from torch.optim import Adam

model,user_encoder,news_encoder,bias_news_encoder,bias_content_scorer,scaler,time_embedding_layer,activity_gater = \
create_pe_model(config, model_config, News, title_word_embedding_matrix, entity_embedding_matrix=None)


# %%

val_metrics_epoch = []
num_epochs = 1
# Step 2: Create your Adam optimizer
optimizer = Adam(model.parameters(), lr=0.0001)

loss_fn = nn.CrossEntropyLoss()

# Step 3: Iterate over the data for the number of epochs
for epoch in range(num_epochs):

    # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
    model.train()
    i = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
    
        # Step 5: Compute the lambda gradient values for the pairwise loss (spedup) with the compute_lambda_i method on the scores and the output labels
        loss = loss_fn(out, y)

        # Step 6: Bacward from the scores with the use of the lambda gradient values
        if loss is not None:
            # torch.autograd.backward(out, loss)
            loss.backward()
            
            # Step 7: Update the weights using the optimizer
            optimizer.step()

    # Step 8: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
    model.eval()

    news_scoring = []
    news_bias_vecs = []
    for x in news_generator:
        news_scoring.append(news_encoder(x))
        news_bias_vecs.append(bias_news_encoder(x))
    news_scoring = torch.cat(news_scoring, dim = 0)
    news_bias_vecs = torch.cat(news_bias_vecs, dim = 0)

    val_user_scoring = []
    for x, y in val_user_generator:
        val_user_scoring.append(user_encoder((x, y)))
    val_user_scoring = torch.cat(val_user_scoring, dim = 0)
    
    val_predicted_activity_gates = activity_gater(val_user_scoring)
    val_predicted_activity_gates = val_predicted_activity_gates[:,0]

    bias_candidate_score = 0

    time_embedding_matrix = time_embedding_layer.weight
    ctr_weight = scaler.scaler[0]
    val_rankings = news_ranking(model_config, ctr_weight, val_predicted_activity_gates, val_user_scoring, news_scoring, 
                               bias_candidate_score, news_bias_vecs, time_embedding_matrix, bias_content_scorer,
                               News,val_impressions)
    
    val_metrics = evaluate_performance(val_rankings,val_impressions)

    print(f"Epoch {epoch+1}/{num_epochs} - , Val Metrics: {val_metrics}")

#     # Step 9: Append the metrics to val_metrics_epoch
    val_metrics_epoch.append(val_metrics)

# %%

news_scoring = []
news_bias_vecs = []
for x in news_generator:
    news_scoring.append(news_encoder(x))
    news_bias_vecs.append(bias_news_encoder(x))
news_scoring = torch.cat(news_scoring, dim = 0)
news_bias_vecs = torch.cat(news_bias_vecs, dim = 0)

test_user_scoring = []
for x, y in test_user_generator:
    test_user_scoring.append(user_encoder((x, y)))
test_user_scoring = torch.cat(test_user_scoring, dim = 0)

test_predicted_activity_gates = activity_gater(test_user_scoring)
test_predicted_activity_gates = test_predicted_activity_gates[:,0]

rankings = news_ranking(model_config,ctr_weight,test_predicted_activity_gates,test_user_scoring,news_scoring,
                                bias_candidate_score,news_bias_vecs,time_embedding_matrix,bias_content_scorer,
                                News,test_impressions)
performance = evaluate_performance(rankings,test_impressions)

cold = []
for TOP_COLD_NUM in [0,1,3,5,]:
    g = evaluate_cold_users(rankings,test_impressions,TestUsers.click,TOP_COLD_NUM)
    cold.append(g)
diversity = []
for TOP_DIVERSITY_NUM in range(1,11):
    div_top = evaluate_diversity_topic_all(TOP_DIVERSITY_NUM,rankings,test_impressions,News,TestUsers)
    div_ilxd = evaluate_density_ILxD(TOP_DIVERSITY_NUM,rankings,test_impressions,news_scoring)
    diversity.append([div_top,div_ilxd])

# %%
print("val metrics", val_metrics_epoch)
print("test metrics", performance)
print("cold", cold)
print("diversity", diversity)

results = {"val metrics": val_metrics_epoch, 
           "test metrics": performance,
           "cold": cold,
           "diversity": diversity}

import json
with open('results.json', 'w') as f:
    json.dump(results, f)

# %% [markdown]
# ## Finished
# ## ---

# %%
# model.compile(loss=['categorical_crossentropy'],
#                   optimizer=Adam(lr=0.0001), 
#                   metrics=['acc'])

# %%
# for i in range(10):

#     model,user_encoder,news_encoder,bias_news_encoder,bias_content_scorer,scaler,time_embedding_layer,activity_gater = create_pe_model(config,model_config,News,title_word_embedding_matrix,News.entity_embedding)
#     model.fit_generator(train_generator,epochs=2)
#     news_scoring = news_encoder.predict_generator(news_generator,verbose=True)
#     user_scoring = user_encoder.predict_generator(test_user_generator,verbose=True)
#     val_user_scoring = user_encoder.predict_generator(val_user_generator,verbose=True)


#     news_bias_vecs = bias_news_encoder.predict_generator(news_generator,verbose=True)

#     if model_config['content'] and not model_config['rece_emb']:
#         bias_candidate_score = bias_content_scorer.predict(news_bias_vecs,batch_size=32,verbose=True)
#         bias_candidate_score = bias_candidate_score[:,0]
#     else:
#         bias_candidate_score = 0

#     ctr_weight = scaler.get_weights()[0][0,0]
#     time_embedding_matrix = time_embedding_layer.get_weights()[0]
    
#     predicted_activity_gates = activity_gater.predict(user_scoring,verbose=True)
#     predicted_activity_gates = predicted_activity_gates[:,0]
    
#     val_predicted_activity_gates = activity_gater.predict(val_user_scoring,verbose=True)
#     val_predicted_activity_gates = val_predicted_activity_gates[:,0]
    
#     rankings = news_ranking(model_config,ctr_weight,predicted_activity_gates,user_scoring,news_scoring,
#                                 bias_candidate_score,news_bias_vecs,time_embedding_matrix,bias_content_scorer,
#                                 News,test_impressions)
    
#     val_rankings = news_ranking(model_config,ctr_weight,val_predicted_activity_gates,val_user_scoring,news_scoring,
#                                bias_candidate_score,news_bias_vecs,time_embedding_matrix,bias_content_scorer,
#                                News,val_impressions)
    
#     performance = evaluate_performance(rankings,test_impressions)
#     val_performance = evaluate_performance(val_rankings,val_impressions)

#     cold = []
#     for TOP_COLD_NUM in [0,1,3,5,]:
#         g = evaluate_cold_users(rankings,test_impressions,TestUsers.click,TOP_COLD_NUM)
#         cold.append(g)
#     diversity = []
#     for TOP_DIVERSITY_NUM in range(1,11):
#         div_top = evaluate_diversity_topic_all(TOP_DIVERSITY_NUM,rankings,test_impressions,News,TestUsers)
#         div_ilxd = evaluate_density_ILxD(TOP_DIVERSITY_NUM,rankings,test_impressions,news_scoring)
#         diversity.append([div_top,div_ilxd])



