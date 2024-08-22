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

## environmental setup

print([np.__version__, pd.__version__])
torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3, suppress=True)

## functions and classes

def make_prior_knowledge_graph(prior_knowledge_matrix):
    d = graphviz.Digraph(engine='dot')

    labels = [f'x{i}' for i in range(prior_knowledge_matrix.shape[0])]
    for label in labels:
        d.node(label, label)

    dirs = np.where(prior_knowledge_matrix > 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        d.edge(labels[from_], labels[to])

    dirs = np.where(prior_knowledge_matrix < 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        if to != from_:
            d.edge(labels[from_], labels[to], style='dashed')
    return d

def get_transformed_data(dim_input, dim_output, data_input, dt, hidden_unit):

    class CustomNN(nn.Module):
        def __init__(self, dt, hidden_unit):
            super(CustomNN, self).__init__()
            if dt=='linear':
                self.nn_reg = nn.Sequential(
                    nn.Linear(dim_input, dim_output),
                )
            else:
                self.nn_reg = nn.Sequential(
                    nn.Linear(dim_input, hidden_unit),
                    nn.Sigmoid(),

                    nn.Linear(hidden_unit, dim_output),
                )

        def forward(self, x):
            output = self.nn_reg(x)
            return output
        
    model = CustomNN(dt, hidden_unit)
    data_output = model(data_input)
    return data_output

def get_generated_data(con, B_true, dcon, n, param_scale, dt, hidden_unit):
    dflat = sum(con)
    G = ig.Graph.Adjacency(B_true.tolist())
    ordered_vertices = G.topological_sorting()  
    assert len(ordered_vertices) == dcon

    dict_new_x = {}
    for v_index in ordered_vertices:
        col = B_true[:, v_index]
        col_sum = np.sum(col, axis=0)
        if col_sum == 0:
            portion_parent = 0
        else:
            dim_output = con[v_index] ## con holds the dimension of output
            dim_input = 0
            data_input = None
            i=0
            for row in col:
                if row == 1:
                    dim_input += con[i]
                    if data_input is None:
                        data_input = dict_new_x[i]
                    else:
                        data_input = torch.cat([data_input, dict_new_x[i]], dim=1) 
                i+=1

            data_output = get_transformed_data(dim_input, dim_output, data_input, dt, hidden_unit)
            portion_parent = data_output.detach()

        portion_noise = torch.randn(n, con[v_index])
        if col_sum == 0:
            new_x = param_scale * portion_noise
        else:
            new_x = param_scale * portion_parent + portion_noise
        dict_new_x[v_index] = new_x

    Xf = dict_new_x[0]
    for i in range(1, dcon):
        Xf = np.hstack((Xf, dict_new_x[i]))
    return Xf

@ray.remote(num_returns=1)
def get_result(
    dt, st, n, d, s0_factor, gt, should_std, trial_no
):
    ## (1a) variable setup
    np.random.seed(123+trial_no) 
    ut.set_random_seed(123+trial_no)                            
    s0 = d * s0_factor
    dcon = d                            
    concept_dim_limit=3
    param_scale = d
    hidden_unit = 100  
    #################################################

    ## (1b) generate a causal graph at random as you have done already (eg. x1->x2) 
    ##     but this time it will represent relations between concepts,
    B_true = ut.simulate_dag(d, s0, gt)                            
    folder_name = str(dt) + '_n_d_s0_gt_sem_' \
                    + str(n) + '_' + str(d) + '_' \
                        + str(s0) + '_' + str(gt) + '_' + str(st)
    folder_path = 'datasets/' + folder_name + '/'
    if os.path.exists(folder_path):
        pass 
    else:
        os.makedirs(folder_path)
    file_name = str(trial_no) + '_W_true.csv'
    file_path = folder_path + file_name
    if os.path.exists(file_path):
        B_true = genfromtxt(file_path, delimiter=',')
    else:                                
        np.savetxt(file_path, B_true, delimiter=',')                            
    #################################################

    ## (2) randomly decide the embedding size of your concepts (eg. dim(x1)=3, dim(x2)=5).
    ##     generate the extended true graph in 'dflat' level.
    concepts = torch.randint(1, concept_dim_limit+1, (dcon,)) 
    concepts = [int(i) for i in concepts]
    print('printing concepts: ', concepts)
    # concepts_fake = torch.randint(1, 2, (dcon,)) 
    # concepts_fake = [int(i) for i in concepts_fake]
    # print('printing concepts_fake: ',concepts_fake )
    dflat = sum(concepts)
    B_true_2 = np.zeros((dflat, dflat))
    end_concept = np.cumsum(concepts)
    for i in range(dcon):
        for j in range(dcon):
            if B_true[i][j]==1:

                if i==0:
                    from_start=0
                else:
                    from_start= end_concept[i-1]
                from_end = end_concept[i]

                if j==0:
                    to_start=0
                else:
                    to_start=end_concept[j-1]
                to_end=end_concept[j]

                for from_x in range(from_start, from_end):
                    for to_x in range(to_start, to_end):
                        B_true_2[from_x][to_x]=1 
    file_name = str(trial_no) + '_W_true_2.csv'
    file_path = folder_path + file_name
    # if os.path.exists(file_path):
        # B_true_2 = genfromtxt(file_path, delimiter=',')
    # else:                                
        # np.savetxt(file_path, B_true_2, delimiter=',')
    np.savetxt(file_path, B_true_2, delimiter=',')
    #################################################

    ## (3) generate a list of neural networks for each effect concept (eg. nn_x2 (input=3, output=5, weights=random), 
    ## (4) generate data for x1 = randn(dim=3) for x2 = nn_x2(x1) + eps*rand(dim=5)

    X = get_generated_data(concepts, B_true, dcon, n, param_scale, dt, hidden_unit)
    file_name = str(trial_no) + '_X.csv'
    file_path = folder_path + file_name
    # if os.path.exists(file_path):
    #     X = genfromtxt(file_path, delimiter=',')
    # else:                                
    #     np.savetxt(file_path, X, delimiter=',')
    np.savetxt(file_path, X, delimiter=',')
    ##
    # X_fake = get_generated_data(concepts_fake, B_true, dcon, n, param_scale, dt, hidden_unit)                                
    # file_name = str(trial_no) + '_X_fake.csv'
    # file_path = folder_path + file_name
    # np.savetxt(file_path, X_fake, delimiter=',')
    if should_std:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)    
    #################################################

    ## (5) run exp
    mask = np.ones((dcon, dcon)) * np.nan
    # n, dflat = X.shape[0], X.shape[1]
    print(concepts, dcon, dflat)
    assert len(concepts) == dcon 
    assert sum(concepts) == dflat
    assert X.shape[1] == dflat    

    ## initializing model and running the optimization
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
        W_notears, res = nonlinear_concept.notears_nonlinear(model, X, lambda1=0.001, lambda2=0.001) ## lambda1=0.01, lambda2=0.01
        # assert ut.is_dag(W_notears)
        # np.savetxt('outputs/W_notears.csv', W_notears, delimiter=',')
        acc = ut.count_accuracy(B_true_2, W_notears != 0)
        print('nCon: ', acc)
        print(W_notears)
        #
        file1 = open('logger.log', 'a+')  
        s1 = "{}, {}, {}, {}, {}, {}, nCon ==> {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            n, d, s0_factor, gt, should_std, trial_no, 
            acc['fdr'], acc['tpr'], acc['fpr'], acc['shd'], acc['nnz']
        )
        file1.writelines(s1)
        file1.close()    
        #
    except Exception as e:
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

    ## initializing model and running the optimization
    try:
        model2 = nonlinear_old.NotearsMLP(dims=[dflat, 10, 1], bias=True)
        W_notears2 = nonlinear_old.notears_nonlinear(model2, X, lambda1=0.001, lambda2=0.001, w_threshold=0.2) ## lambda1=0.01, lambda2=0.01, w_threshold=0.3
        # assert ut.is_dag(W_notears2)
        # np.savetxt('outputs/W_notears2.csv', W_notears2, delimiter=',')
        acc2 = ut.count_accuracy(B_true_2, W_notears2 != 0)
        print('n_reg', acc2)
        #
        file1 = open('logger.log', 'a+')  
        s1 = "{}, {}, {}, {}, {}, {}, nReg ==> {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            n, d, s0_factor, gt, should_std, trial_no, 
            acc2['fdr'], acc2['tpr'], acc2['fpr'], acc2['shd'], acc2['nnz']
        )                            
        file1.writelines(s1)
        file1.close()
        #
    except Exception as e:
        acc2 = {
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
        (acc2['fdr'], acc2['tpr'], acc2['fpr'], acc2['shd'], acc2['nnz']), 
    ]

if __name__=='__main__':

    ## variables

    #
    list_dt_st = [('nonlinear', 'mlp')] ## [('nonlinear', 'mlp'), ('linear', 'mlp')]
    list_n = [200, 1000] ## [200, 1000]
    list_d = [10, 20] ## [10, 20]
    list_s0_factor = [1, 4] ## [1, 4]
    list_gt = ['ER', 'SF'] ## ['ER', 'SF']
    list_should_std = [False, True] ## [False, True]
    n_trials = 50 ## 10
    #
    
    ## experiments

    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=56) ## detects automatically: num_cpus=64

    for dt, st in list_dt_st:
        for n in list_n:
            for d in list_d:
                for s0_factor in list_s0_factor:
                    for gt in list_gt:
                        for should_std in list_should_std:

                            list_result_id = []
                            for trial_no in range(n_trials):
                                result_id = get_result.remote(
                                    dt, st, n, d, s0_factor, gt, should_std, trial_no
                                )
                                list_result_id.append(result_id)
                            list_result = ray.get(list_result_id)

                            d_result = {}
                            for trial_no in range(n_trials):
                                d_result[(n, d, s0_factor, gt, should_std, trial_no, 'nCon')] = list_result[trial_no][0]
                                d_result[(n, d, s0_factor, gt, should_std, trial_no, 'nReg')] = list_result[trial_no][1]

                            with open(
                                'datasets/d_result_' + str(n) + '_' + str(d) + '_' + str(s0_factor) + '_' + str(gt) + '_' + str(should_std) + '.pickle', 'wb'
                            ) as handle: 
                                pk.dump(d_result, handle, protocol=pk.HIGHEST_PROTOCOL)
