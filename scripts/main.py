import argparse
import os
from typing import  Tuple
from tqdm import tqdm
import random

from pso_pipline import pso_pipeline
from sos_pipline import sos_pipeline

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
hyper_parameters= ((500,10,30),(300,30,100),(160,55,170),(160,55,200),(120,80,300),(100,100,500),(120,55,None))

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

all_categories_BBC_Sport = []
###

def apply_file(res_dict:dict,file_name :str):
    import pickle 
    os.system(f"touch {file_name}.pkl")
    print("file created")
    with open(file_name + ".pkl" , "wb") as f :
        pickle.dump(res_dict , f)
        print("file saved")


def main_runner(DataSet:str,NUM_CLUSTERS:int,NUM_DATA:int,algorithm:str,HP:Tuple[tuple],file_name_to_save:str,NIA:int = 2):

    results  = []
    for hy_pa in tqdm(HP):
                
                if DataSet == "20 News Group":    
                    my_category = random.sample(all_categories_20news,NUM_CLUSTERS)
                elif DataSet == "BBC Sport":
                    my_category = random.sample(all_categories_BBC_Sport,NUM_CLUSTERS)
                else : 
                    raise ValueError("Dataset name is wrong!!! or does not exsit")
               
                for i in range(NIA):
                    if algorithm == "PSO":
                        g_best  ,start , elapsed  = pso_pipeline(dataset=DataSet , categories=my_category,K=NUM_CLUSTERS,NUM_SAMPLES = NUM_DATA,
                                                        max_feature=hy_pa[2],epoch=hy_pa[0],pop_size=hy_pa[1])
                    elif algorithm == "SOS":
                        g_best  ,start , elapsed  = sos_pipeline(dataset=DataSet , categories=my_category,K=NUM_CLUSTERS,NUM_SAMPLES = NUM_DATA,
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




if __name__ == "__main__":
    main_runner(DataSet=args.d,NUM_CLUSTERS=args.cluster,NUM_DATA=args.n,algorithm=args.a,
                HP=hyper_parameters,file_name_to_save=args.f)