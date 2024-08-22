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

## change_0 ===========================================================================================
# df_1 = pd.read_csv('datasets/adult_data.csv', header=None)
# df_2 = pd.read_csv('datasets/adult_test.csv', header=None)
# df_3 = pd.concat([df_1, df_2])
# for col in list(df_3.columns):
#     df_3 = df_3[df_3[col] != ' ?'] 

# list_col = list(df_3.columns)
# list_remove = [2,4,10,11] ## fnlwgt, education-num, capital-gain, capital-loss
# list_col_valid = []
# for col in list_col:
#     if col not in list_remove:
#         list_col_valid.append(col)
# print(list_col_valid)

# df_4 = df_3[list_col_valid]
# df_4 = df_4.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})
# print(df_4.shape)

# df_4.to_csv('datasets/adult_processed.csv', index=False, header=None) 

# change_0 ===========================================================================================

## change_1 ============================================================================================
## data and causal graph

df_x = pd.read_csv('datasets/dutch.csv', header=None)
df_cg = pd.read_excel(open('datasets/dutch.xlsx', 'rb'), index_col=0)


print(df_x.shape)
df_x.head(2)

print(df_cg.shape)
df_cg.head(2)
## change_1 ============================================================================================

#### experiment

## functions and classes 
@ray.remote(num_returns=1)
def get_result(data_x, data_cg, should_std, trial_no):
        
    ## 1
    np.random.seed(123+trial_no) 
    ut.set_random_seed(123+trial_no) 

    ## 2
    ## change_2 ==========================================================================================    

    ## all are categorical
    ## 0. sex: -
    ## 1. age: -
    ## 2. household_position: -
    ## 3. household_size: -
    ## 4. prev_residence_place: -
    ## 5. citizenship: -
    ## 6. country_birth: -
    ## 7. edu_level: -
    ## 8. economic_status: -
    ## 9. cur_eco_activity: -
    ## 10. Marital_status: -     
    ## 11. occupation: -


    concepts = []    
    list_index_continuous = [] ## 
    Xcon, B_true = df_x.values, df_cg.values
    Xflat = None
    for i in range(Xcon.shape[1]):
        Xlocal = Xcon[:, i].reshape(-1, 1)    
        if i in list_index_continuous:
            Xlocalflat = Xlocal
        else:
            Xlocalflat = LabelEncoder().fit_transform(Xlocal.ravel()).reshape(-1,1)
            # Define the size of the vocabulary and the size of the embedding vectors
            vocab_size = np.max(Xlocalflat) + 1
            embedding_dim = math.floor(math.log2(vocab_size) + 1)
            # Create an instance of the Embedding class
            embedding_layer = nn.Embedding(vocab_size, embedding_dim)    
            embedding_vectors = embedding_layer(torch.tensor(Xlocalflat))
            Xlocalflat = embedding_vectors.reshape(-1, embedding_dim).detach().numpy()    

        concepts.append(Xlocalflat.shape[1])
        if Xflat is None:
            Xflat = Xlocalflat
        else:
            Xflat = np.hstack((Xflat, Xlocalflat))    
    
    n, d = Xcon.shape
    s0 = sum(sum(B_true))   
    print('concepts ======> ', concepts)
    
    
    
    dcon, dflat = len(concepts), sum(concepts)
    print(n, d, s0, concepts, dcon, dflat, Xcon.shape, B_true.shape, Xflat.shape)


    ## 3
    if should_std:
        # scalerCon = StandardScaler().fit(Xcon)
        # Xcon = scalerCon.transform(Xcon) ## this includes string values, so can't transform    
        scalerFlat = StandardScaler().fit(Xflat)
        Xflat = scalerFlat.transform(Xflat)    
    # Xcon, Xflat = Xcon.astype('float32'), Xflat.astype('float32')
    Xflat = Xflat.astype('float32')
    ## change_2 ==========================================================================================    
        

    ## 4
    mask = np.ones((dcon, dcon)) * np.nan
    print(concepts, dcon, dflat)
    assert len(concepts) == dcon 
    assert sum(concepts) == dflat
    assert Xcon.shape[1] == dcon        
    assert Xflat.shape[1] == dflat    

    ## initializing model and running the optimizationportion_parent
    try:
        metainfo = {}
        metainfo['dflat'] = dflat
        metainfo['dcon'] = dcon
        metainfo['concepts'] = concepts                            
        model = nonlinear_concept.NotearsMLP(
            dims=[dflat, 10, 1], bias=True,
            mask=mask, w_threshold=0.2, learned_model=None, ## w_threshold=0.3
            metainfo=metainfo
        )
        W_notears, res = nonlinear_concept.notears_nonlinear(
            model, Xflat, lambda1=0.001, lambda2=0.001,
            h_tol=1e-4, rho_max=1e+8
        ) ## lambda1=0.01, lambda2=0.01, h_tol=1e-8, rho_max=1e+16
        # assert ut.is_dag(W_notears)
        # np.savetxt('outputs/W_notears.csv', W_notears, delimiter=',')
        acc = ut.count_accuracy(B_true, W_notears != 0)
        print('nCon: ', acc)
        print(W_notears)
        #
        file1 = open('logger.log', 'a+')  
        s1 = "{}, {}, nCon ==> {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            should_std, trial_no, 
            acc['fdr'], acc['tpr'], acc['fpr'], acc['shd'], acc['nnz']
        )
        file1.writelines(s1)
        file1.close()    
        #
    except Exception as e:
        print('========================================', e)
        acc = {
            'fdr': '-',
            'tpr': '-',
            'fpr': '-',
            'shd': '-',
            'nnz': '-'
        }
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
            model3, Xflat, lambda1=0.001, lambda2=0.001, w_threshold=0.2,
            h_tol=1e-4, rho_max=1e+8
        ) ## lambda1=0.01, lambda2=0.01, w_threshold=0.3, h_tol=1e-8, rho_max=1e+16
        W_notears3 = conv_flat_to_con(W_notears3, concepts)
        # assert ut.is_dag(W_notears3)
        # np.savetxt('outputs/W_notears3.csv', W_notears3, delimiter=',')
        acc3 = ut.count_accuracy(B_true, W_notears3 != 0)
        print('nRegFlat', acc3)
        print(W_notears3)        
        #
        file1 = open('logger.log', 'a+')  
        s1 = "{}, {}, nRegFlat ==> {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            should_std, trial_no, 
            acc3['fdr'], acc3['tpr'], acc3['fpr'], acc3['shd'], acc3['nnz']
        )                            
        file1.writelines(s1)
        file1.close()
        #
    except Exception as e:
        acc3 = {
            'fdr': '-',
            'tpr': '-',
            'fpr': '-',
            'shd': '-',
            'nnz': '-'
        }
        file1 = open('logger.log', 'a+')  
        s1 = "Error ==> {}\n".format(e)
        file1.writelines(s1)
        file1.close()                    


    #################################################
 
    
    return [
        (acc['fdr'], acc['tpr'], acc['fpr'], acc['shd'], acc['nnz']), 
        (acc3['fdr'], acc3['tpr'], acc3['fpr'], acc3['shd'], acc3['nnz']),        
    ]


if __name__=='__main__':

    ## variables
    list_should_std = [False, True]
    n_trials = 50
    
    ## variables            
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=56) ## detects automatically: num_cpus=64

    ## experiments
    for should_std in list_should_std:
        list_result_id = []
        for trial_no in range(n_trials):
            result_id = get_result.remote(
                df_x, df_cg, should_std, trial_no
            )
            list_result_id.append(result_id)
        list_result = ray.get(list_result_id)

        d_result = {}
        for trial_no in range(n_trials):
            d_result[(should_std, trial_no, 'nCon')] = list_result[trial_no][0]
            d_result[(should_std, trial_no, 'nRegFlat')] = list_result[trial_no][1]                                

        with open(
            'datasets/d_result_' + str(should_std) + '.pickle', 'wb'
        ) as handle: 
            pk.dump(d_result, handle, protocol=pk.HIGHEST_PROTOCOL)
      
    
