import pandas as pd
import os
import copy
import ast
from tqdm import tqdm
import gensim
import langid
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy
import re
import string
import math
from gensim import corpora, models
import pickle
from sklearn.model_selection import KFold
import statistics
from gensim.models.coherencemodel import CoherenceModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

def lit_eval(x):
    if x!='':
        x=ast.literal_eval(x)
    return x

def lang_check(word_list):
    for word in word_list:
        if 'ー' in word:
            continue
        res=re.search("([^\x00-\x7F])+",word)
        if res is not None:
            return False
    return True

def lower_case(y):
    if len(y)>0 and y!='':
        y=[z.lower() for z in y]
    return y

def remove_punc(li):
    if len(li)>0 and li!='':
        li=[z.translate(str.maketrans('', '', string.punctuation)) for z in li]
    return li

def check_for_irrelevant(li):
    new_li=[]
    if len(li)>0 and li!='':
        for x in range(len(li)):
            if 'ー' in li[x]:
                new_li.append(li[x].replace('ー',''))
            if not li[x].startswith('corona'):
                new_li.append(li[x])
    return new_li

def check_lang(text):
    return langid.classify(text)[0]

hashtag_counts = 0
users_hashtags={}
day_wise={}
filespath = '/data/Coronavirus-Tweets/daily-hashtags'
files = os.listdir(filespath)
for file in tqdm(files):
    day = file.split('.')[0]
    if day not in day_wise:
        day_wise[day]=[]
    df = pickle.load(open(os.path.join(filespath,file),'rb'))
    df['hashtags'] = df['hashtags'].apply(lower_case)
    for i in range(len(df)):
        if df['user'].iloc[i] not in users_hashtags:
            users_hashtags[df['user'].iloc[i]] = []

        users_hashtags[df['user'].iloc[i]].extend(df['hashtags'].iloc[i])
        hashtag_counts+=len(df['hashtags'].iloc[i])

        day_wise[day].extend(df['hashtags'].iloc[i])

user_df = pd.DataFrame(users_hashtags.items(),columns=['users','documents'])

tqdm.pandas()
user_df['docs'] = user_df['documents'].tolist()
user_df['docs'] = user_df['docs'].astype('str')
user_df['lang'] = user_df['docs'].progress_apply(check_lang)
new_user_df=user_df[user_df['lang']=='en']

new_user_df['documents']=new_user_df['documents'].apply(check_for_irrelevant)

docs_df=new_user_df
docs_df['lang']=docs_df['documents'].apply(lang_check)
irrelevant = docs_df[docs_df['lang']==False]
docs_df=docs_df[docs_df['lang']==True].reset_index()
docs=docs_df['documents'].tolist()

dictionary = gensim.corpora.Dictionary(docs)
#Remove hashtags present in less than 15 documents and ones present in more than three-quarters of the documents,
#then keep top 100000 hashtags
dictionary.filter_extremes(no_below=10, no_above=0.75, keep_n=100000)

#construct a bag_of_words corpus of hashtags for each document
bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

#Create a tf-idf model on the bag of words corpus and apply it.
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

coherence_values = {}
model_list = {}

for i in tqdm([10,20,50,100,200]):
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=50, workers=47,random_state=23,minimum_probability=0)
    model_list[i]=lda_model
    coherencemodel = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
    coherence_values[i]=coherencemodel.get_coherence()
best_model = Counter(coherence_values).most_common()[0][0]
lda_model = model_list[best_model]

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

user_features={}
for corpus_index in tqdm(range(len(bow_corpus))):
    user_features[docs_df['users'].iloc[corpus_index]]=[]
    results = lda_model.get_document_topics(bow_corpus[corpus_index],minimum_probability=0)
    for r in results:
        user_features[docs_df['users'].iloc[corpus_index]].append(r[1])

user_features_df=pd.DataFrame(user_features.items(),columns=['user','feature_vector'])
mbfc_scores=pd.read_csv('/data/domain-score.csv')
subdf=user_features_df[user_features_df.user.isin(mbfc_scores['user'].tolist())]

combined_df=subdf.merge(mbfc_scores,on='user')
combined_df['feature_vector'] = combined_df['feature_vector'].apply(lambda y: np.nan if len(y)==0 else y)
combined_df=combined_df[combined_df['feature_vector'].notna()]

hard_sci=[]
hard_politics=[]
hard_moderacy=[]

q1_sci=combined_df['science'].quantile(0.33)
q3_sci=combined_df['science'].quantile(0.67)

q1_pol=combined_df['political'].quantile(0.33)
q3_pol=combined_df['political'].quantile(0.67)

q1_mod=combined_df['moderacy'].quantile(0.33)
q3_mod=combined_df['moderacy'].quantile(0.67)

for i in range(len(combined_df)):
    if combined_df['science'].iloc[i]<=q1_sci:
        hard_sci.append(-1.0)
    elif combined_df['science'].iloc[i]>=q3_sci:
        hard_sci.append(1.0)
    else:
        hard_sci.append(-100)
    if combined_df['political'].iloc[i]<=q1_pol:
        hard_politics.append(-1.0)
    elif combined_df['political'].iloc[i]>=q3_pol:
        hard_politics.append(1.0)
    else:
        hard_politics.append(-100)
    if combined_df['moderacy'].iloc[i]<=q1_mod:
        hard_moderacy.append(-1.0)
    elif combined_df['moderacy'].iloc[i]>=q3_mod:
        hard_moderacy.append(1.0)
    else:
        hard_moderacy.append(-100)
combined_df['hard_sci']=hard_sci
combined_df['hard_politics']=hard_politics
combined_df['hard_moderacy']=hard_moderacy

##Science LDA Results
tmp = combined_df[combined_df['hard_sci']!=-100]
X_new=np.asarray(tmp['feature_vector'].tolist())
Y_new=np.asarray(tmp['hard_sci'].tolist())

print("Science Results")
acc=[]
prec=[]
rec=[]
f1=[]
kfold = KFold(5, True, 1)
for train, test in kfold.split(X_new):
    print(len(X_new[train]),len(X_new[test]))

    X_train=X_new[train]
    y_train=Y_new[train]

    X_test=X_new[test]
    y_test=Y_new[test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(random_state=23,max_iter=700).fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    print("Accuracy",accuracy_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred))
    print("Recall",recall_score(y_test,y_pred))
    print("F1-Score",f1_score(y_test,y_pred))
    print()

    acc.append(accuracy_score(y_test,y_pred))
    prec.append(precision_score(y_test,y_pred))
    rec.append(recall_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred))
print("Mean accuracy-Science",statistics.mean(acc))
print("Mean precision-Science",statistics.mean(prec))
print("Mean recall-Science",statistics.mean(rec))
print("Mean F1-Score-Science",statistics.mean(f1))

##Political LDA Results
tmp = combined_df[combined_df['hard_politics']!=-100]
X_new=np.asarray(tmp['feature_vector'].tolist())
Y_new=np.asarray(tmp['hard_politics'].tolist())

print("Politics Results")
acc=[]
prec=[]
rec=[]
f1=[]
kfold = KFold(5, True, 1)
for train, test in kfold.split(X_new):
    print(len(X_new[train]),len(X_new[test]))

    X_train=X_new[train]
    y_train=Y_new[train]

    X_test=X_new[test]
    y_test=Y_new[test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(random_state=23,max_iter=700).fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print("Accuracy",accuracy_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred))
    print("Recall",recall_score(y_test,y_pred))
    print("F1-Score",f1_score(y_test,y_pred))
    print()

    acc.append(accuracy_score(y_test,y_pred))
    prec.append(precision_score(y_test,y_pred))
    rec.append(recall_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred))
print("Mean accuracy-Political",statistics.mean(acc))
print("Mean precision-Political",statistics.mean(prec))
print("Mean recall-Political",statistics.mean(rec))
print("Mean F1-Score-Political",statistics.mean(f1))

##Moderacy LDA Results
tmp = combined_df[combined_df['hard_moderacy']!=-100]
X_new=np.asarray(tmp['feature_vector'].tolist())
Y_new=np.asarray(tmp['hard_moderacy'].tolist())

print("Moderacy Results")
acc=[]
prec=[]
rec=[]
f1=[]
kfold = KFold(5, True, 1)
for train, test in kfold.split(X_new):
    print(len(X_new[train]),len(X_new[test]))

    X_train=X_new[train]
    y_train=Y_new[train]

    X_test=X_new[test]
    y_test=Y_new[test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(random_state=23,max_iter=700).fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print("Accuracy",accuracy_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred))
    print("Recall",recall_score(y_test,y_pred))
    print("F1-Score",f1_score(y_test,y_pred))
    print()

    acc.append(accuracy_score(y_test,y_pred))
    prec.append(precision_score(y_test,y_pred))
    rec.append(recall_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred))
print("Mean accuracy-Moderacy",statistics.mean(acc))
print("Mean precision-Moderacy",statistics.mean(prec))
print("Mean recall-Moderacy",statistics.mean(rec))
print("Mean F1-Score-Moderacy",statistics.mean(f1))


##You can add a simple snippet to classify whole set of users.
