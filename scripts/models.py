from typing import List, Tuple
import numpy as np
from mealpy import Problem
from mealpy.utils.space import BaseVar
from numpy import ndarray

from mealpy.utils.space import FloatVar

def generate(K:int,dataset:None):
    '''
    Args:
    K = number of clusters
    '''
    data_sorted = np.sort(dataset, axis=0)
  
    lbs = data_sorted[0:K,:] 
    # lbs = np.ones(shape= (K * dataset.shape[1])) * data_sorted[0]

    ubs = data_sorted[-K:,:]
    # ubs = np.ones(shape= (K * dataset.shape[1])) * data_sorted[-1]
    #number of features
    m = dataset.shape[1]
    bounds = [FloatVar(lb = [lbs[j][i] for i in range(m)] , ub=[ubs[j][i] for i in range(m)])
             for j in range(K)]
    
    # bounds = FloatVar(lb =lbs , ub=ubs)
    # FloatVar(lb =lbs , ub=ubs)
    
    return bounds


class Data_Clustering_Purity_Obj_Func(Problem):

    def __init__(self, bounds: List | Tuple | ndarray | BaseVar,
                 K : int ,name : str = "data_clustering",
                 target=None,
                 dataset = None,
                 minmax: str = "max",**kwargs) -> None:
        '''
        Args :
        K = number of cluster centers
        dataset  = ndarray (without labels)
        target = list of data labels
        Returns :
        value of objective function
        '''
        self.name = name
        self.K = K
        self.dataset = dataset
        self.target =  target
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution : ndarray) -> float:
        # solution == organism == individual population
  
        
        distance_from_centers = np.array([[np.linalg.norm(row  - cluster_center) 
                                  for cluster_center in np.array(np.reshape(solution,(self.K,self.dataset.shape[1]))) ]
                                  for row in self.dataset]) #c1r1 , c2r1 , cnr1 
                                                            #..............ckrn  k = number of clusters  , n = number of data samples
        
        clustered = [[] for i in range(self.K)]

        for i , row_dis in enumerate(distance_from_centers):
            label = np.where(row_dis==np.min(row_dis))[0][0]
            clustered[label].append(self.target[i])
        # in the enhances version tha labels dirctly append to clusters with target and index 
        
        #number of data
        n = self.dataset.shape[0]   
        
        #this lines counts which label in each cluster is more and sum together them      
        sum = 0
        for li in clustered :
            sum += max(li.count(i) for i in range(self.K))
        
        return sum / n
    

class Data_Clustering_Distance_Obj_Func(Problem):
    def __init__(self, bounds: List | Tuple | ndarray | BaseVar,
                 K : int ,
                 name : str = "data_clustering",
                 dataset = None,
                 minmax: str = "min",
                 **kwargs) -> None:
        '''
        Args :
        K = number of cluster centers
        dataset  = ndarray (without labels)
        target = list of data labels
        Returns :
        value of objective function
        '''
        self.name = name
        self.K = K
        self.dataset = dataset
       
        super().__init__(bounds, minmax, **kwargs)
    

    
    def obj_func(self, solution : ndarray) -> float:
        # solution == organism == individual population

        #number of features in dataset
        # m = self.dataset.shape[1]

        norms = [[np.linalg.norm(row  - cluster_center)
                  for cluster_center in np.array(np.reshape(solution,(self.K,self.dataset.shape[1]))) ] 
                  for row in self.dataset]
        valu_obj =  np.sum(

            np.fromiter((np.min(norms[i]) for i in range(len(norms))),dtype=np.float32)
        )
        
        return valu_obj

