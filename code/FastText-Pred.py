import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

df1 = pickle.load(open('../models/en-all-users-mbfc-fasttext-embed_df_1.pkl','rb'))
df2 = pickle.load(open('../models/en-all-users-mbfc-fasttext-embed_df_2.pkl','rb'))
df1['feature_vector']=df1.values.tolist()
df2['feature_vector']=df2.values.tolist()
df1 = df1.reset_index()
df2 = df2.reset_index()

df1 = df1.rename(columns={'index': 'user'})
df2 = df2.rename(columns={'index': 'user'})

df1=df1[['user','feature_vector']]
df2=df2[['user','feature_vector']]

#Overall
new_df=df1.append(df2,ignore_index=True)
mbfc_scores=pd.read_csv('../data/domain-score.csv')
subdf=new_df[new_df.user.isin(mbfc_scores['user'].tolist())]


combined_df=subdf.merge(mbfc_scores,on='user')
#Consider the top and bottom 30% of the data
s1 = combined_df['science'].quantile(0.3)
s2 = combined_df['science'].quantile(0.7)
p1 = combined_df['political'].quantile(0.3)
p2 = combined_df['political'].quantile(0.7)
m1 = combined_df['moderacy'].quantile(0.3)
m2 = combined_df['moderacy'].quantile(0.7)


hard_sci=[]
hard_politics=[]
hard_moderacy=[]

for i in range(len(combined_df)):
    if combined_df['science'].iloc[i]<=s1:
        hard_sci.append(0)
    elif combined_df['science'].iloc[i]>=s2:
        hard_sci.append(1.0)
    else:
        hard_sci.append(-100)
    if combined_df['political'].iloc[i]<=p1:
        hard_politics.append(0)
    elif combined_df['political'].iloc[i]>=p2:
        hard_politics.append(1.0)
    else:
        hard_politics.append(-100)
    if combined_df['moderacy'].iloc[i]<=m1:
        hard_moderacy.append(0)
    elif combined_df['moderacy'].iloc[i]>=m2:
        hard_moderacy.append(1.0)
    else:
        hard_moderacy.append(-100)
combined_df['hard_sci']=hard_sci
combined_df['hard_politics']=hard_politics
combined_df['hard_moderacy']=hard_moderacy


tmp = combined_df[combined_df['hard_sci']!=-100]
X_new=tmp['feature_vector']
Y_new=tmp['hard_sci']

X_train, X_test, y_train, y_test = train_test_split(X_new, Y_new,test_size=0.33,shuffle=True,random_state=23)

clf = LogisticRegression(random_state=23,max_iter=700).fit(np.asarray(X_train.tolist()), y_train.tolist())
y_pred=clf.predict(np.asarray(X_test.tolist()))
print("Science-Predictions")
print("Accuracy",accuracy_score(np.asarray(y_test.tolist()),y_pred))
print("Precision",precision_score(np.asarray(y_test.tolist()),y_pred))
print("Recall",recall_score(np.asarray(y_test.tolist()),y_pred))
print("F1-Score",f1_score(np.asarray(y_test.tolist()),y_pred))
tn, fp, fn, tp = confusion_matrix(np.asarray(y_test.tolist()), y_pred).ravel()

all_X = new_df['feature_vector'].tolist()
all_Y_pred = clf.predict(all_X)
new_df['science']=list(all_Y_pred)

tmp = combined_df[combined_df['hard_politics']!=-100]
X_new=tmp['feature_vector']
Y_new=tmp['hard_politics']

X_train, X_test, y_train, y_test = train_test_split(X_new, Y_new,test_size=0.33,shuffle=True,random_state=23)
print()
print("Political-Predictions")
clf = LogisticRegression(random_state=23,max_iter=700).fit(np.asarray(X_train.tolist()), y_train.tolist())
y_pred=clf.predict(np.asarray(X_test.tolist()))
print("Accuracy",accuracy_score(np.asarray(y_test.tolist()),y_pred))
print("Precision",precision_score(np.asarray(y_test.tolist()),y_pred))
print("Recall",recall_score(np.asarray(y_test.tolist()),y_pred))
print("F1-Score",f1_score(np.asarray(y_test.tolist()),y_pred))
tn, fp, fn, tp = confusion_matrix(np.asarray(y_test.tolist()), y_pred).ravel()


all_X = new_df['feature_vector'].tolist()
all_Y_pred = clf.predict(all_X)
new_df['political']=list(all_Y_pred)

tmp = combined_df[combined_df['hard_moderacy']!=-100]
X_new=tmp['feature_vector']
Y_new=tmp['hard_moderacy']

X_train, X_test, y_train, y_test = train_test_split(X_new, Y_new,test_size=0.33,shuffle=True,random_state=23)

print()
print("Moderacy-Predictions")
clf = LogisticRegression(random_state=23,max_iter=700).fit(np.asarray(X_train.tolist()), y_train.tolist())
y_pred=clf.predict(np.asarray(X_test.tolist()))
print("Accuracy",accuracy_score(np.asarray(y_test.tolist()),y_pred))
print("Precision",precision_score(np.asarray(y_test.tolist()),y_pred))
print("Recall",recall_score(np.asarray(y_test.tolist()),y_pred))
print("F1-Score",f1_score(np.asarray(y_test.tolist()),y_pred))
tn, fp, fn, tp = confusion_matrix(np.asarray(y_test.tolist()), y_pred).ravel()

all_X = new_df['feature_vector'].tolist()
all_Y_pred = clf.predict(all_X)
new_df['moderacy']=list(all_Y_pred)

del new_df['feature_vector']
new_df.to_csv('../results/Final-Predictions.csv',index=False)
