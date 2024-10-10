import argparse
import os
from typing import  Tuple
from tqdm import tqdm
import random

from pso_pipline import pso_pipeline_20news , pso_pipeline_bbc_sport
from sos_pipline import sos_pipeline_news , sos_pipeline_bbc_sport

parser = argparse.ArgumentParser(description="Runner of Tests ")
### Parameters that set in shell
parser.add_argument("--cluster",type=int, help="number of clusters (int)")
parser.add_argument("--num_data",type=int, help="number of sampls (int)")
parser.add_argument("--file_name",type=str, help="file_name_to_save (str)")
parser.add_argument("--algorithm_name",type=str, help="Algorith name : PSO | SOS (str)")
parser.add_argument("--dataset_name",type=str, help="Dataset name : 20 News Group | BBC Sport (srt)")
### 

# parser.add_argument("epoch" ,type=int, help="number of itrations(int)")
# parser.add_argument("pop_size",type=int, help="number of population(int)")

args = parser.parse_args()

### Parameters that set in script
hyper_parameters= ((100,10,30),(100,20,60),(100,30,100))

all_categories_20news = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

###

def apply_file(res_dict:dict,file_name :str):
    import pickle 
    os.system(f"touch {file_name}.pkl")
    print("file created")
    with open(file_name + ".pkl" , "wb") as f :
        pickle.dump(res_dict , f)
        print("file saved")


def main_runner_20news(NUM_CLUSTERS:int,NUM_DATA:int,algorithm:str,HP:Tuple[tuple],file_name_to_save:str,NIA:int = 2):

    results  = []
    for hy_pa in tqdm(HP):
                  
                my_category = random.sample(all_categories_20news,NUM_CLUSTERS)
               
                for i in range(NIA):
                    if algorithm == "PSO":
                        g_best  ,start , elapsed  = pso_pipeline_20news(categories=my_category,K=NUM_CLUSTERS,NUM_SAMPLES = NUM_DATA,
                                                        max_feature=hy_pa[2],epoch=hy_pa[0],pop_size=hy_pa[1])
                    elif algorithm == "SOS":
                        g_best  ,start , elapsed  = sos_pipeline_news(categories=my_category,K=NUM_CLUSTERS,NUM_SAMPLES = NUM_DATA,
                                                        max_feature=hy_pa[2],epoch=hy_pa[0],pop_size=hy_pa[1])
                    else :
                         raise ValueError("Algorithm name is wrong!!! or does not exsit")
                    
                    # Save results
                    results.append({
                        "hy_pa" : hy_pa , 
                        "g_best" : g_best,
                        "start_index" : start ,
                        "categories" : my_category,
                        "elapsed" : elapsed,
                        "run" : i
                    })

                    print(f"hyper parameters {hy_pa} category {my_category}")

    apply_file(results,file_name=file_name_to_save)


def main_runner_bbc_sport(NUM_CLUSTERS:int,NUM_DATA:int,algorithm:str,HP:Tuple[tuple],file_name_to_save:str,NIA:int = 2):
    results  = []
    for hy_pa in tqdm(HP):               
                for i in range(NIA):
                    if algorithm == "PSO":
                        g_best, elapsed  = pso_pipeline_bbc_sport(K=NUM_CLUSTERS,NUM_SAMPLES = NUM_DATA,
                                                        max_feature=hy_pa[2],epoch=hy_pa[0],pop_size=hy_pa[1])
                    elif algorithm == "SOS":
                        g_best,elapsed  = sos_pipeline_bbc_sport(K=NUM_CLUSTERS,NUM_SAMPLES = NUM_DATA,
                                                        max_feature=hy_pa[2],epoch=hy_pa[0],pop_size=hy_pa[1])
                    else :
                         raise ValueError("Algorithm name is wrong!!! or does not exsit")
                    
                    # Save results
                    results.append({
                        "hy_pa" : hy_pa , 
                        "g_best" : g_best,
                        "elapsed" : elapsed,
                        "run" : i
                    })

                    print(f"hyper parameters {hy_pa}")

    apply_file(results,file_name=file_name_to_save)

    return results

if __name__ == "__main__":
    if args.dataset_name == "20 News Group":
        main_runner_20news(NUM_CLUSTERS=args.cluster,NUM_DATA=args.num_data,algorithm=args.algorithm_name,
                    HP=hyper_parameters,file_name_to_save=args.file_name)
    elif args.dataset_name == "BBC Sport" :
        main_runner_bbc_sport(NUM_CLUSTERS=args.cluster,NUM_DATA=args.num_data,algorithm=args.algorithm_name,
                    HP=hyper_parameters,file_name_to_save=args.file_name)
         