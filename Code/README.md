# PP-Rec
- Code of our PP-Rec reproduction

# Data Preparation
- Codes in this project can be used on the Ebnerd dataset (https://recsys.eb.dk/dataset/) for experiments
- The DEMO dataset is saved and already preprocessed in this repo
- All recommendation data are stored in data-root-path
- We used the glove.840B.300d embedding vecrors in https://nlp.stanford.edu/projects/glove/
- The embedding file should be stored in embedding\_path\glove.840B.300d.txt
- The image embedding pkl file for training on those should be saved in embedding\_path\image_embeddings.pkl
- The meta data such as entities be stored in KG\_root\_path, this data can be extracted from the ebnerd dataset using the reformat_data.ipynb notebook

# Code Files
- utils.py: containing some util functions
- preprocess.py: containing functions to preprocess data
- NewsContent.py: Class for loading and managing news data
- UserContent.py: Class for loading and managing user data
- PEGenerator.py: data generator for model training and evaluation
- models.py: containing basic models such as Attention network
- Encoders.py: containing encoders or predictors in our model
- Main.ipynb: containing codes for model training and evaluation
