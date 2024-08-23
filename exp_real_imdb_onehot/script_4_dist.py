## import

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from scipy.special import expit as sigmoid
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
# import graphviz
import notears.utils as ut
from notears import nonlinear_concept, nonlinear_old
import igraph as ig
# import lingam
# from lingam.utils import make_prior_knowledge, make_dot
import ray
import pickle as pk
from scipy.special import expit as sigmoid
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import math

## environmental setup

print([np.__version__, pd.__version__])
torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3, suppress=True)

@ray.remote(num_returns=1)
def get_result(df_x, opt):
        
    ## 1
    should_std, val_lambda, w_threshold = opt[0], opt[1], opt[2]    
    np.random.seed(123) 
    ut.set_random_seed(123) 

    ## 2

    # 'budget', 
    # 'w0', 'w1','w2', 'w3','w4', 'w5','w6', 'w7','w8', 'w9','w10', 'w11',   
    # 'd0','d1','d2', 'd3','d4', 'd5','d6', 'd7','d8', 'd9',   
    # 'p0','p1','p2', 'p3','p4', 'p5','p6', 'p7','p8', 'p9','p10', 'p11', 'p12',    
    # 'c0','c1','c2', 'c3','c4', 'c5','c6', 'c7','c8', 'c9','c10', 'c11', 'c12', 'c13',    
    # 'g0','g1','g2', 'g3','g4',
    # 'week_of_year',
    # 'imdb_user_rating',
    # 'revenue'        
    # concepts = [1, 12, 10, 13, 14, 5, 1, 1, 1] 
    concepts = [1, 14, 5, 1, 1, 1]   
    Xflat = df_x.values
    
    ## 3
    if should_std:
        scalerFlat = StandardScaler().fit(Xflat)
        Xflat = scalerFlat.transform(Xflat)    
    Xflat = Xflat.astype('float32')
    n, dflat = Xflat.shape
    dcon = len(concepts)
    
    ## 4
    mask = np.ones((dcon, dcon)) * np.nan
    print(concepts, dcon, dflat)
    assert len(concepts) == dcon 
    assert sum(concepts) == dflat
    assert Xflat.shape[1] == dflat    

    ## initializing model and running the optimizationportion_parent
    try:
        metainfo = {}
        metainfo['dflat'] = dflat
        metainfo['dcon'] = dcon
        metainfo['concepts'] = concepts                            
        model = nonlinear_concept.NotearsMLP(
            dims=[dflat, 10, 1], bias=True,
            mask=mask, w_threshold=w_threshold, learned_model=None, ## w_threshold=0.3
            metainfo=metainfo
        )
        W_notears, res = nonlinear_concept.notears_nonlinear(
            model, Xflat, lambda1=val_lambda, lambda2=val_lambda,
            h_tol=1e-8, rho_max=1e+18
        ) ## lambda1=0.01, lambda2=0.01, h_tol=1e-8, rho_max=1e+16
        # assert ut.is_dag(W_notears)
        np.savetxt('outputs/W_con_' + str(should_std) + str(val_lambda) + str(w_threshold) + '.csv', W_notears, delimiter=',')
        print('W_con', W_notears)
        #
        #
    except Exception as e:
        print('========================================', e)
        file1 = open('logger.log', 'a+')  
        s1 = "Error ==> {}\n".format(e)
        file1.writelines(s1)
        file1.close()                    


    ## initializing model and running the optimizaportion_parenttion
    def conv_flat_to_con(A, concepts):

        ##
        A = np.abs(A) ## in the optimization this works on square matrix, so there we don't need to abs it
        dflat = sum(concepts)
        dcon = len(concepts)
        Arow = np.zeros((dcon,dflat))
        Ad = np.zeros((dcon,dcon))
        end_concept = np.cumsum(concepts)

        ##
        start_i = 0
        for i in range(dcon):
            end_i = end_concept[i]
            Arow[i,:] = (A[start_i:end_i,:].sum(axis=0))/(end_i-start_i)
            start_i = end_i
        start_i = 0
        for i in range(dcon):
            end_i = end_concept[i]
            Ad[:,i] = (Arow[:,start_i:end_i].sum(axis=1))/(end_i-start_i)
            start_i = end_i

        ##
        new_adj_mat = np.zeros((dcon,dcon))
        for i in range(dcon):
            for j in range(dcon):
                if Ad[i][j] != 0:
                    new_adj_mat[i][j] = 1

        return new_adj_mat

    try:
        model3 = nonlinear_old.NotearsMLP(dims=[dflat, 10, 1], bias=True)
        W_notears3 = nonlinear_old.notears_nonlinear(
            model3, Xflat, lambda1=val_lambda, lambda2=val_lambda, w_threshold=w_threshold,
            h_tol=1e-8, rho_max=1e+18
        ) ## lambda1=0.01, lambda2=0.01, w_threshold=0.3, h_tol=1e-8, rho_max=1e+16
        W_notears3 = conv_flat_to_con(W_notears3, concepts)
        # assert ut.is_dag(W_notears3)
        np.savetxt('outputs/W_flat_' + str(should_std) + str(val_lambda) + str(w_threshold) + '.csv', W_notears3, delimiter=',')
        print('W_flat', W_notears3)        
        #
        #
    except Exception as e:
        file1 = open('logger.log', 'a+')  
        s1 = "Error ==> {}\n".format(e)
        file1.writelines(s1)
        file1.close()                    

    return 0

if __name__=='__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=56) ## detects automatically: num_cpus=64
    

    list_option = [
        (True, 0.005, 0.3),
        (True, 0.005, 0.2),                               
        (True, 0.001, 0.3),
        (True, 0.001, 0.2),                
        (True, 0.0005, 0.3),
        (True, 0.0005, 0.2),                                
    ]
    df_x = pd.read_csv('datasets/movie_processed_1.csv')
    df_x = df_x[[
#         'budget', 
#         'w0', 'w1','w2', 'w3','w4', 'w5','w6', 'w7','w8', 'w9','w10', 'w11',   
#         'd0','d1','d2', 'd3','d4', 'd5','d6', 'd7','d8', 'd9',   
#         'p0','p1','p2', 'p3','p4', 'p5','p6', 'p7','p8', 'p9','p10', 'p11', 'p12',    
#         'c0','c1','c2', 'c3','c4', 'c5','c6', 'c7','c8', 'c9','c10', 'c11', 'c12', 'c13',    
#         'g0','g1','g2', 'g3','g4',
#         'week_of_year',
#         'imdb_user_rating',
#         'revenue'        
        
        'budget', 
#         'w0', 'w1','w2', 'w3','w4', 'w5','w6', 'w7','w8', 'w9','w10', 'w11',   
#         'd0','d1','d2', 'd3','d4', 'd5','d6', 'd7','d8', 'd9',   
#         'p0','p1','p2', 'p3','p4', 'p5','p6', 'p7','p8', 'p9','p10', 'p11', 'p12',    
        'c0','c1','c2', 'c3','c4', 'c5','c6', 'c7','c8', 'c9','c10', 'c11', 'c12', 'c13',    
        'g0','g1','g2', 'g3','g4',
        'week_of_year',
        'imdb_user_rating',
        'revenue'        
    ]]
        
    
    list_result_id = []
    for opt in list_option:
        result_id = get_result.remote(
            df_x, opt
        )
        list_result_id.append(result_id)
    list_result = ray.get(list_result_id)

