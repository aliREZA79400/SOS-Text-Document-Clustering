import os
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import time
from sklearn.datasets import  fetch_20newsgroups
####
from mealpy.bio_based import SOS
from preprocess import apply_preprocess_on_dataset
from models import generate
from models import Data_Clustering_Purity_Obj_Func
# from models import Data_Clustering_Purity_Obj_Func


BASE_DIR = os.getcwd()

def sos_pipeline(dataset=None,categories=None,K:int=None,NUM_SAMPLES:int=None,
                   max_feature:int=None,epoch:int=None,pop_size:int=None):

    if dataset == "20 News Group" :
        X = fetch_20newsgroups(data_home=BASE_DIR, subset="all", categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'), download_if_missing=False, return_X_y=False)
    elif dataset == "BBC Sport" :
        pass

    n = len(X.data) # n = 4875
    start_index = random.sample(range(0,n-NUM_SAMPLES),1)  # like 4775
    preprocess_data = apply_preprocess_on_dataset(X.data[start_index[0]:start_index[0] +NUM_SAMPLES])
    vectorizer_tfidf = TfidfVectorizer(max_features=max_feature)
    vector_dataset= vectorizer_tfidf.fit_transform(preprocess_data)

    targets = X.target[start_index[0]:start_index[0] +NUM_SAMPLES]

    bound = generate(K=K,dataset=vector_dataset.toarray())

    ### define problem with class {log training process}
    problem_ins = Data_Clustering_Purity_Obj_Func(bounds=bound,
                                                K=K,
                                                target=targets,
                                                dataset=vector_dataset.toarray())

    ### build model (instance)
    model = SOS.OriginalSOS(epoch=epoch,pop_size=pop_size)

    ### train model  with solve ( training modes )
    start_time = time.time()
    g_best  = model.solve(problem=problem_ins)
    end_time = time.time()
    elapsed  = end_time - start_time

    return g_best  , start_index , elapsed