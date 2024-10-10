import os
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from scipy.io import mmread
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

def sos_pipeline_news(categories=None,K:int=None,NUM_SAMPLES:int=None,
                   max_feature:int=None,epoch:int=None,pop_size:int=None):


    X = fetch_20newsgroups(data_home=BASE_DIR, subset="all", categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'), download_if_missing=False, return_X_y=False)

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

    return g_best  , start_index[0] , elapsed


######
mtx_file_path = "/media/alireza/SSD1/arshad_hosh/Thesis/ThesisCode/bbcsport/bbcsport.mtx"
# targets
trg = []
with open("/media/alireza/SSD1/arshad_hosh/Thesis/ThesisCode/bbcsport/bbcsport.classes" , "r") as f :
    for line in f :
        trg.append(int(line.split()[1]))

def sos_pipeline_bbc_sport(K:int=None,NUM_SAMPLES:int=None,       
                   max_feature:int=None,epoch:int=None,pop_size:int=None):
    

    # Read the mtx file into a scipy sparse matrix
    numpy_mtx = mmread(mtx_file_path).toarray().transpose()

    data_with_target =   list(zip(numpy_mtx , trg))
    if K == 5 :
        data_with_target = random.sample(data_with_target,NUM_SAMPLES)
    else :
        # class_labals = random.sample(range(5),K)
        class_labals = [0,1]
        data_with_target = [i for i in data_with_target if i[1] in class_labals]
        data_with_target = random.sample(data_with_target,NUM_SAMPLES)

    sampled_data , sampled_target = zip(*data_with_target)
    sampled_data = np.array(list(sampled_data))

    # Create a TfidfTransformer object
    tfidf_transformer = TfidfTransformer()

    # Fit the transformer to the count matrix and transform it to a TF-IDF matrix
    tfidf = tfidf_transformer.fit_transform(sampled_data)
    svd = TruncatedSVD(n_components=max_feature)
    X_new = svd.fit_transform(tfidf.toarray())

    bound = generate(K=K,dataset=X_new)

    ### define problem with class {log training process}
    problem_ins = Data_Clustering_Purity_Obj_Func(bounds=bound,
                                                  K=K,
                                                  target=list(sampled_target),
                                                  dataset=X_new)


    ### build model (instance)
    model = SOS.OriginalSOS(epoch=epoch,pop_size=pop_size)

    ### train model  with solve ( training modes )
    start_time = time.time()
    g_best  = model.solve(problem=problem_ins)
    end_time = time.time()
    elapsed  = end_time - start_time

    return g_best, elapsed


