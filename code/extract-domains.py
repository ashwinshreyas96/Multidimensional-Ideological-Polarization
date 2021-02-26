from ast import literal_eval
import pandas as pd
import os
from tqdm import tqdm
import ast
import sys


filespath = sys.argv[1] #provide path to rehydrated day wise tweets in .csv format filtered by location in US
savepath = '../data/user-domains-extracted'

if not os.path.exists(savepath):
    os.makedirs(savepath)

folders = os.listdir(filespath)
dict_of_dfs={}

for folder in folders:
    files = os.listdir(os.path.join(filespath,folder))
    for file in tqdm(files):
        date = file.split('.')[0].split('clean-')[1].split('-')
        date = '-'.join(date[:3])
        df = pd.read_csv(os.path.join(filespath,folder,file))
        df = df[['screen_name','lang','urls_list','rt_urls_list','rt_screen']]
        df = df[df['lang']=='en']


        df1 = df[['screen_name','urls_list']]
        df2 = df[['rt_screen','rt_urls_list']]


        df2 = df2.rename(columns={'rt_screen': 'screen_name'})
        df2 = df2.rename(columns={'rt_urls_list': 'urls_list'})


        df1 = df1[df1['screen_name'].notna()]
        df2 = df2[df2['screen_name'].notna()]

        new_df = pd.DataFrame(columns=['screen_name','urls_list'])

        new_df = new_df.append(df1,ignore_index=True)
        new_df = new_df.append(df2,ignore_index=True)

        new_df = new_df[new_df['urls_list'].notna()]
        new_df = new_df[new_df['screen_name'].notna()]

        new_df = new_df[new_df.screen_name.isin(rel_users)]

        new_df['urls_list'] = new_df['urls_list'].astype('str')
        new_df['urls_list'] = new_df['urls_list'].apply(ast.literal_eval)
        new_df['len'] = new_df['urls_list'].apply(lambda x: len(x))

        new_df = new_df[new_df['len']>0]

        new_df['links'] = new_df['urls_list'].apply(lambda x: x[0]['expanded_url'])

        new_df = new_df.groupby('screen_name')['links'].apply(list).reset_index(name='links')

        if date not in dict_of_dfs:
            dict_of_dfs[date]=new_df
        else:
            dict_of_dfs[date] = dict_of_dfs[date].append(new_df,ignore_index=True)

    for date in dict_of_dfs:
        tmp = dict_of_dfs[date].groupby('screen_name').agg({'links':'sum'}).reset_index()
        tmp.to_pickle(os.path.join(savepath,date+'.pkl'),index=False)
    dict_of_dfs={}
