import pandas as pd
import os
from igraph import *
import statistics
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from sklearn.semi_supervised import LabelPropagation
from sklearn import preprocessing
from collections import Counter
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from functools import reduce
import multiprocessing
import time
import sys
num_cores = multiprocessing.cpu_count()
print("numCores = " + str(num_cores))

def create_graph(unweighted_edges, boolean):
    g = Graph.TupleList(G, directed=boolean)
    g.es["weight"] = 1
    g.simplify(combine_edges={"weight": "sum"})
    print('Graph Created')
    return(g)

def descriptive_stats(g):
    print("Maximum indegree for a node : ",max(g.indegree()))
    print("Maximum outdegree for a node : ",max(g.outdegree()))
    print("Maximum retweeted handle : ",g.vs.select(_indegree = max(g.indegree()))["name"])
    print("Maximum retweeter handle : ",g.vs.select(_outdegree = max(g.outdegree()))["name"])
    print("Number of vertices in retweet network : ",g.vcount())
    print("Number of edes in retweet network : ",g.ecount())
    print("Number of cliques in retweet network : ",g.clique_number())


def graph_nodes_df(g):
    nodes_list = []
    for i in range(len(g.vs)):
        nodes_list.append(g.vs[i].attributes())
    df =  pd.DataFrame(nodes_list)
    return df

rdf=pd.read_csv(sys.argv[1]) #Pass retweet network in form of a .csv file containing edges
rdf = rdf[rdf['rt_screen'].notna()]
rdf = rdf[rdf['screen_name'].notna()]
G = [tuple(x) for x in rdf[['screen_name', 'rt_screen']].values]
g_directed = create_graph(G, True)
g_undirected = create_graph(G, False)

nodes_df = graph_nodes_df(g_undirected)
nodes_df = nodes_df.rename(columns={'name': 'user'})
nodes_list = nodes_df['user'].values.tolist()

seeds_df=pd.read_csv('../data/Seeds-LPA-Science.csv') #Can do same for Politics/Moderacy. 
seeds_df['polarization'] = seeds_df['polarization'].map({'Pro-Science': 1, 'Conspiracy-Pseudoscience': 2})

def classify_label_propagation(seeds_df, nodes_df, split_num):
    t_ideo_X = np.array(seeds_df['user'])
    t_ideo_y = np.array(list(seeds_df['polarization']))
    t_ideo_skf = StratifiedKFold(n_splits= split_num)
    t_ideo_skf.get_n_splits(t_ideo_X, t_ideo_y)
    ct=0
    merged_df_list = []
    predicted_labels_list = []
    for train_index, test_index in t_ideo_skf.split(t_ideo_X, t_ideo_y):
        ct+=1
        t_ideo_equiv = { 1: True, 2: True, -1: False}
        equiv_t_ideo = {0: 1, 1: 2, -1: -1}

        df_train = pd.DataFrame({'user': t_ideo_X[train_index], 'label': t_ideo_y[train_index]})
        df_test = pd.DataFrame({'user': t_ideo_X[test_index], 'label': t_ideo_y[test_index]})
        #print("Test-Set:",len(df_test))
        df_test.to_csv('round'+str(ct)+'.csv')
        #print(len(nodes_df))
        df_train =  nodes_df.merge(df_train, how='left', on='user').fillna(-1)
        duplicate_bool = df_train.duplicated(subset=['user'], keep='first')
        duplicate = df_train.loc[duplicate_bool == True]
        df_train['fixed'] = df_train['label'].map(t_ideo_equiv)
        df_train=df_train.drop_duplicates(subset=['user'], keep='first').reset_index()
        #print(len(df_train))
        label_prop = Graph.community_label_propagation(g_undirected , weights = 'weight',initial = df_train['label'],fixed = df_train['fixed'])

        for n in range(0,len(label_prop)):
            print('Community #', n, 'size:', len(label_prop[n]))

        df_train['predicted_label'] = label_prop.membership
        df_train['predicted_label'] = df_train['predicted_label'].map(equiv_t_ideo)
        predicted_labels_list.append(df_train)

        df =  pd.merge(left=df_test, right= df_train, how='left',
                                     left_on='user', right_on='user').dropna(how='any')
        merged_df_list.append(df)
    return merged_df_list, predicted_labels_list

def getScore(df_list, measure):
    score_list = []
    for df in df_list:
        score = measure(df['label_x'], df['predicted_label'], average='micro')
        score_list.append(score)
    return sum(score_list)/len(df_list)

def getCommunityMembership(community_list):
    for i in range(0,len(community_list)):
        print(Counter(community_list[i]['predicted_label']))

def getFinalLabelsForNodes(list_of_dfs):
    df_list = []
    for df in list_of_dfs:
        df_list.append(df[['user', 'predicted_label']])
    df = reduce(lambda left,right: pd.merge(left,right,on='user'), df_list)
    df = df.replace(-1, 0)
    df['label'] = df.iloc[:,1:6].sum(axis = 1)/5 #col numbers between in the bracket and divided by the number of splits
    df = df[['user', 'label']]
    df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1 if 0 < x <= 1.5 else 2)
    df['label'] = df['label'].map({0: 'no_label', 1: 'Pro-Science', 2: 'Conspiracy-Pseudoscience'})
    return df

label_prop_df_list = classify_label_propagation(seeds_df,nodes_df,5)
user_label_df = getFinalLabelsForNodes(label_prop_df_list[1])

user_label_df=pd.read_csv('../results/LPA-Science-Results.csv')
