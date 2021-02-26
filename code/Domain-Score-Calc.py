import pandas as pd
import os
import pickle as pkl
import tldextract
from tqdm import tqdm
import statistics
from collections import Counter
import numpy as np


## SCIENCE - DOMAIN SCORE CALCULATION

filespath='../data/user-domains-extracted'
# Load our seed set of pay-level domains sourced from Media Bias Fact Check
scidf=pd.read_csv('../data/pro-science.csv')
antiscidf=pd.read_csv('../data/conspiracy-pseudoscience.csv')

#We have 150 pro-science PLDs and 450 anti-science PLDs


#Function to extract domain names from URLs posted by users on their tweets
def extract(url):
    uri = tldextract.extract(url)
    uri = '.'.join(uri[1:3])
    return uri


#Perform the same extraction of domain names on the seed set so as to ensure consistency
for i in tqdm(range(len(scidf['domain']))):
    uri = tldextract.extract( scidf['domain'].iloc[i])
    scidf['domain'].iloc[i]='.'.join(uri[1:3])

for i in tqdm(range(len(antiscidf['domain']))):
    uri = tldextract.extract(antiscidf['domain'].iloc[i])
    antiscidf['domain'].iloc[i]='.'.join(uri[1:3])


valid_domains=[]
valid_domains.extend(scidf['domain'].tolist())
valid_domains.extend(antiscidf['domain'].tolist())
valid_domains=list(set(valid_domains))


sci_doms_count=None
antisci_doms_count=None

#Function to filter out irrelevant PLDs and
#count the frequency of these PLDs to consider an equal number of them on both ends of the spectrum
def clean(links):
    global sci_doms_count
    global antisci_doms_count
    links = [extract(x) for x in links]
    links = [x for x in links if x in valid_domains]
    sci_doms = [x for x in links if x in scidf['domain'].tolist()]
    antisci_doms = [x for x in links if x in antiscidf['domain'].tolist()]
    if sci_doms_count is None:
        sci_doms_count=Counter(sci_doms)
    else:
        sci_doms_count+=Counter(sci_doms)

    if antisci_doms_count is None:
        antisci_doms_count=Counter(antisci_doms)
    else:
        antisci_doms_count+=Counter(antisci_doms)

    if len(links)>0:
        return links
    else:
        return None


list_of_files=os.listdir(filespath)
list_of_files.sort()
links_generated_by_user={}
for file in tqdm(list_of_files):
    df=pkl.load(open(os.path.join(filespath,file),'rb'))
    df['links']=df['links'].apply(clean)
    df=df[df['links'].notna()]
    curr_dict = dict(zip(df['screen_name'], df['links']))
    for user in curr_dict:
        if user not in links_generated_by_user:
            links_generated_by_user[user]=[]
        links_generated_by_user[user].extend(curr_dict[user])


minimum_among_two = np.min([len(scidf),len(antiscidf)])
sci_list=sci_doms_count.most_common()[:minimum_among_two]
sci_list = [x[0] for x in sci_list]
antisci_list=antisci_doms_count.most_common()[:minimum_among_two]
antisci_list = [x[0] for x in antisci_list]


user_polarity_list={}
for user in links_generated_by_user:
    user_links=links_generated_by_user[user]
    if len(user_links)>0:
        user_polarity_list[user]=[]
        for link in user_links:
            if link in sci_list:
                user_polarity_list[user].append(1)
            elif link in antisci_list:
                user_polarity_list[user].append(-1)

#Compute the mean domain score per user.
user_means={}
user_link_counts={}
user_pro_links={}
user_anti_links={}
for user in tqdm(user_polarity_list):
    if len(user_polarity_list[user])>2:
        user_link_counts[user]=len(user_polarity_list[user])
        user_pro_links[user]=user_polarity_list[user].count(1)
        user_anti_links[user]=user_polarity_list[user].count(-1)
        user_mean=statistics.mean(user_polarity_list[user])
        user_means[user]=user_mean

## POLITICAL - DOMAIN SCORE CALCULATION


leftdf=pd.read_csv('../data/left.csv') #1
rightdf=pd.read_csv('../data/right.csv')#-1


for i in tqdm(range(len(leftdf['domain']))):
    uri = tldextract.extract(leftdf['domain'].iloc[i])
    leftdf['domain'].iloc[i]='.'.join(uri[1:3])


for i in tqdm(range(len(rightdf['domain']))):
    uri = tldextract.extract(rightdf['domain'].iloc[i])
    rightdf['domain'].iloc[i]='.'.join(uri[1:3])

valid_domains_lr=[]
valid_domains_lr.extend(leftdf['domain'].tolist())
valid_domains_lr.extend(rightdf['domain'].tolist())
valid_domains_lr=list(set(valid_domains_lr))


left_doms_count=None
right_doms_count=None


def clean_lr(links):
    global left_doms_count
    global right_doms_count
    links = [extract(x) for x in links]
    links = [x for x in links if x in valid_domains_lr]

    left_doms = [x for x in links if x in leftdf['domain'].tolist()]
    right_doms = [x for x in links if x in rightdf['domain'].tolist()]
    if left_doms_count is None:
        left_doms_count=Counter(left_doms)
    else:
        left_doms_count+=Counter(left_doms)

    if right_doms_count is None:
        right_doms_count=Counter(right_doms)
    else:
        right_doms_count+=Counter(right_doms)

    if len(links)>0:
        return links
    else:
        return None

list_of_files=os.listdir(filespath)
list_of_files.sort()
links_generated_by_user_lr={}
for file in tqdm(list_of_files):
    df=pkl.load(open(os.path.join(filespath,file),'rb'))
    df['links']=df['links'].apply(clean_lr)
    df=df[df['links'].notna()]
    curr_dict = dict(zip(df['screen_name'], df['links']))
    for user in curr_dict:
        if user not in links_generated_by_user_lr:
            links_generated_by_user_lr[user]=[]
        links_generated_by_user_lr[user].extend(curr_dict[user])


minimum_among_two = np.min([len(leftdf),len(rightdf)])
left_list=left_doms_count.most_common()[:minimum_among_two]
left_list = [x[0] for x in left_list]
right_list=right_doms_count.most_common()[:minimum_among_two]
right_list = [x[0] for x in right_list]


user_polarity_list_lr={}
for user in links_generated_by_user_lr:
    user_links=links_generated_by_user_lr[user]
    if len(user_links)>0:
        user_polarity_list_lr[user]=[]
        for link in user_links:
            if link in left_list:
                user_polarity_list_lr[user].append(-1)
            elif link in right_list:
                user_polarity_list_lr[user].append(1)

user_means_lr={}
user_link_counts_lr={}
user_left_links={}
user_right_links={}
for user in tqdm(user_polarity_list_lr):
    if len(user_polarity_list_lr[user])>2:
        user_link_counts_lr[user]=len(user_polarity_list_lr[user])
        user_right_links[user]=user_polarity_list_lr[user].count(1)
        user_left_links[user]=user_polarity_list_lr[user].count(-1)
        user_mean=statistics.mean(user_polarity_list_lr[user])
        #user_means_lr[user]=user_right_links[user]/(user_right_links[user]+user_left_links[user])
        user_means_lr[user]=user_mean


## MODERACY - DOMAIN SCORE CALCULATION

intensedf = pd.read_csv('../data/intense.csv') #-1
centerdf = pd.read_csv('../data/moderate.csv')#1

for i in tqdm(range(len(intensedf['domain']))):
    uri = tldextract.extract(intensedf['domain'].iloc[i])
    intensedf['domain'].iloc[i]='.'.join(uri[1:3])

for i in tqdm(range(len(centerdf['domain']))):
    uri = tldextract.extract(centerdf['domain'].iloc[i])
    centerdf['domain'].iloc[i]='.'.join(uri[1:3])

valid_domains_ic=[]
valid_domains_ic.extend(intensedf['domain'].tolist())
valid_domains_ic.extend(centerdf['domain'].tolist())
valid_domains_ic=list(set(valid_domains_ic))

intense_doms_count=None
center_doms_count=None

def clean_ic(links):
    global intense_doms_count
    global center_doms_count

    links = [extract(x) for x in links]
    links = [x for x in links if x in valid_domains_ic]

    intense_doms = [x for x in links if x in intensedf['domain'].tolist()]
    center_doms = [x for x in links if x in centerdf['domain'].tolist()]
    if intense_doms_count is None:
        intense_doms_count=Counter(intense_doms)
    else:
        intense_doms_count+=Counter(intense_doms)

    if center_doms_count is None:
        center_doms_count=Counter(center_doms)
    else:
        center_doms_count+=Counter(center_doms)

    if len(links)>0:
        return links
    else:
        return None

list_of_files=os.listdir(filespath)
list_of_files.sort()
links_generated_by_user_ic={}
for file in tqdm(list_of_files):
    df=pkl.load(open(os.path.join(filespath,file),'rb'))
    df['links']=df['links'].apply(clean_ic)
    df=df[df['links'].notna()]
    curr_dict = dict(zip(df['screen_name'], df['links']))
    for user in curr_dict:
        if user not in links_generated_by_user_ic:
            links_generated_by_user_ic[user]=[]
        links_generated_by_user_ic[user].extend(curr_dict[user])


minimum_among_two = np.min([len(intensedf),len(centerdf)])
intense_list=intense_doms_count.most_common()[:minimum_among_two]
intense_list = [x[0] for x in intense_list]
center_list=center_doms_count.most_common()[:minimum_among_two]
center_list = [x[0] for x in center_list]


user_polarity_list_ic={}
for user in tqdm(links_generated_by_user_ic):
    user_links=links_generated_by_user_ic[user]
    if len(user_links)>0:
        user_polarity_list_ic[user]=[]
        for link in user_links:
            if link in intense_list:
                user_polarity_list_ic[user].append(-1)
            elif link in center_list:
                user_polarity_list_ic[user].append(1)

user_means_ic={}
user_link_counts_ic={}
user_intense_links={}
user_moderate_links={}
for user in tqdm(user_polarity_list_ic):
    if len(user_polarity_list_ic[user])>2:
        user_link_counts_ic[user]=len(user_polarity_list_ic[user])
        user_moderate_links[user]=user_polarity_list_ic[user].count(1)
        user_intense_links[user]=user_polarity_list_ic[user].count(-1)
        user_mean=statistics.mean(user_polarity_list_ic[user])
        user_means_ic[user]=user_mean


mod_df = pd.DataFrame(user_means_ic.items(),columns=['user','moderacy'])
sci_df = pd.DataFrame(user_means.items(),columns=['user','science'])
pol_df = pd.DataFrame(user_means_lr.items(),columns=['user','political'])

res = sci_df.merge(pol_df,on='user')
res = res.merge(mod_df,on='user')
res.to_csv('../data/domain-score.csv',index=False)
