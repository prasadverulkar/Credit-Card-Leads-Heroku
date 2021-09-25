#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier,plot_importance, plot_tree
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


%%time
train= reduce_mem_usage(pd.read_csv(r"E:\Data science\DataHack\Analytics Vidhya\Job_a_thon_28052021\train_s3TEQDk.csv"))
test= reduce_mem_usage(pd.read_csv(r"E:\Data science\DataHack\Analytics Vidhya\Job_a_thon_28052021\test_mSzZ8RL.csv"))
print("Shape of train set: ",train.shape)
print("Shape of test set: ",test.shape)

# In[4]:


train.head()


# In[5]:


train.describe(include = 'all')


# In[6]:


train.info()


# In[7]:


train.Is_Lead.value_counts()


# ## Data Preprocessing
# ### Missing Values

# In[8]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# In[10]:


train.Credit_Product.fillna("Missing",inplace = True)
test.Credit_Product.fillna("Missing",inplace = True)


# ### Transformation

# In[11]:


f , ax = plt.subplots(1,2,figsize = (10,5))
sns.histplot(data = train, x = "Avg_Account_Balance", log_scale=True,ax = ax[0])
sns.histplot(data = test, x = "Avg_Account_Balance", log_scale=True,ax = ax[1])


# In[12]:


train["Avg_Account_Balance"] = np.log(train['Avg_Account_Balance'])
test["Avg_Account_Balance"] = np.log(train['Avg_Account_Balance'])


# ### Feature Engineering

# In[13]:


train["Is_train"] = 1
test["Is_train"] = 0
full_df = train.append(test)
full_df.shape


# In[14]:


full_df.head()


# In[15]:


full_df["Is_Salaried_40_65"] = np.where(((full_df["Age"]>=37.5)|(full_df["Age"]<=65))&(full_df["Occupation"]=="Salaried"),1,0)


# In[16]:


full_df.head()


# ### Categorical Encodings

# In[17]:


factcols = list(full_df.select_dtypes(include = 'object').columns)
factcols.remove('ID')
factcols.remove('Region_Code')


# In[18]:


factcols


# In[19]:


# Label Encode all columns except Region_Code

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

full_df[factcols] = full_df[factcols].apply(le.fit_transform)



# In[20]:

train_new = full_df[full_df.Is_train == 1]
train_new.drop(columns = ["ID","Is_train"],inplace= True)
test_new = full_df[full_df.Is_train == 0]
test_new.drop(columns = ["ID","Is_train","Is_Lead"],inplace= True)
train_new.head()


# In[21]:

import pickle
pickle.dump(le, open('le_encode.pkl','wb')) 
train_x = train_new.loc[:,train_new.columns != 'Is_Lead']
train_y = train_new.loc[:,['Is_Lead']]


# In[22]:


# Target Encoding

from sklearn.model_selection import StratifiedKFold

new_train = train.copy()
fold = StratifiedKFold(n_splits = 5,shuffle = False)
global_mean = train["Is_Lead"].mean()
alpha = 100
new = pd.DataFrame()
abc = np.zeros(len(train))

for train_ind, val_ind in fold.split(train_x, train_y):
    tr, val = train_new.iloc[train_ind], train_new.iloc[val_ind]
    
    target_encode = tr.groupby(["Region_Code"])["Is_Lead"].transform("mean")
    freq = tr.groupby(["Region_Code"])["Is_Lead"].transform("count")
    reg = ( target_encode*freq + global_mean*alpha)  / (freq + alpha)
    temp = tr[["Region_Code"]]
    temp['reg'] = reg
    tar_enc = temp.groupby("Region_Code").mean()
    
    abc[val_ind] = val["Region_Code"].map(tar_enc['reg'])

new['target_enc_col'] = abc
new['Region_Code'] = train_new['Region_Code']
value = new.groupby(['Region_Code'])["target_enc_col"].mean()
    
train_new['Region_Code'] = train_new['Region_Code'].map(value)
test_new['Region_Code'] = test_new['Region_Code'].map(value)
train_new.fillna(global_mean, inplace = True), test_new.fillna(global_mean, inplace = True)

pickle.dump(value, open('tar_enc.pkl', 'wb'))

# In[23]:


train.shape
test.shape


# ### Frequency Encoding

# In[24]:


# fr = train_new.groupby('Region_Code').size() / len(train_new)
# train_new.Region_Code = train_new.Region_Code.apply(lambda x: fr[x])


# ## Sampling

# In[25]:


train_x = train_new.loc[:,train_new.columns != 'Is_Lead']
train_y = train_new.loc[:,['Is_Lead']]


# In[26]:


tr_x,vd_x,tr_y,vd_y = train_test_split(train_x,train_y,test_size = 0.3, random_state=498, stratify = train_y.values)


# In[27]:


tr_x.shape

vd_x.shape

tr_y.shape

vd_y.shape


# In[28]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini',max_depth = 4)
dtree.fit(tr_x,tr_y)
y_pred = dtree.predict(vd_x) ; y_pred_train = dtree.predict(tr_x)
tab = confusion_matrix(y_pred , vd_y) ; tab1 = confusion_matrix(y_pred_train , tr_y)

from sklearn.metrics import roc_auc_score
ROCAUC_score = roc_auc_score(vd_y , dtree.predict_proba(vd_x)[:, 1])
ROCAUC_score_train = roc_auc_score(tr_y , dtree.predict_proba(tr_x)[:, 1])
print("ROC_AUC_Score_Validation:",ROCAUC_score,"ROC_AUC_Score_train:",ROCAUC_score_train,"\nConfusion Matrix for Validation:\n",tab)


# In[29]:


#pd.DataFrame({"Columns":tr_x.columns,"FI":dtree.feature_importances_}).sort_values(by = "FI",ascending = False)


# In[30]:


#from sklearn.tree import plot_tree


# In[31]:


#fn = list(tr_x.columns)
#cn = ["0","1"]
#
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (22,20), dpi=300)
#
#plot_tree(dtree,feature_names = fn,fontsize = 6 ,
#               class_names=cn,
#               filled = True);
#fig.savefig('imagename.png')


# In[32]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini',max_depth = 5)
rf.fit(tr_x,tr_y)
y_pred = rf.predict(vd_x) ; y_pred_train = rf.predict(tr_x)
tab = confusion_matrix(y_pred , vd_y) ; tab1 = confusion_matrix(y_pred_train , tr_y)

from sklearn.metrics import roc_auc_score
ROCAUC_score = roc_auc_score(vd_y , rf.predict_proba(vd_x)[:, 1])
ROCAUC_score_train = roc_auc_score(tr_y , rf.predict_proba(tr_x)[:, 1])
print("ROC_AUC_Score:",ROCAUC_score,"ROC_AUC_Score_train:",ROCAUC_score_train,"\nConfusion Matrix for Validation:\n",tab)


# In[33]:


# XGBoost

from xgboost import XGBClassifier
xgc = XGBClassifier(n_jobs = -1 )
xgc.fit(tr_x,tr_y,eval_metric='logloss')
y_pred = xgc.predict(vd_x) ; y_pred_train = xgc.predict(tr_x)
tab = confusion_matrix(y_pred , vd_y) ; tab1 = confusion_matrix(y_pred_train , tr_y)

from sklearn.metrics import roc_auc_score
ROCAUC_score = roc_auc_score(vd_y , xgc.predict_proba(vd_x)[:, 1])
print("ROC_AUC_Score:",ROCAUC_score,"\nConfusion Matrix for Validation:\n",tab)


# In[34]:


# LightGBM
from lightgbm import LGBMClassifier,plot_importance,plot_tree

lgb = LGBMClassifier()
lgb.fit(tr_x,tr_y)
y_pred = lgb.predict(vd_x) ; y_pred_train = lgb.predict(tr_x)
tab = confusion_matrix(y_pred , vd_y) ; tab1 = confusion_matrix(y_pred_train , tr_y)

from sklearn.metrics import roc_auc_score
ROCAUC_score = roc_auc_score(vd_y, lgb.predict_proba(vd_x)[:, 1])
print("ROC_AUC_Score:",ROCAUC_score,"\nConfusion Matrix for Validation:\n",tab)


# In[35]:


#plot_importance(lgb, importance_type='gain',height = 0.8, dpi = 90)


# ## Hyperparameter Tuning using Genetic Algorithm

# In[36]:


# Number of trees in LightGBM
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_bin = [int(x) for x in np.linspace(10,100,10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10,50,10)]
# Minimum number of samples required to split a node
num_leaves = [10,15,37,20]
# Minimum number of samples required at each leaf node
subsample = [0.6,0.7,0.75]
# Learning Rate
learning_rate = [0.01,0.03,0.04]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_bin': max_bin,
               'max_depth': max_depth,
               'num_leaves': num_leaves,
               'subsample': subsample,
              'learning_rate':learning_rate}
print(param)


# In[37]:


#from tpot import TPOTClassifier


# In[357]:


#tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
#                                 verbosity= 2, early_stop= 12,
#                                 config_dict={'lightgbm.LGBMClassifier': param}, 
#                                 cv = 5, scoring = 'roc_auc')
#tpot_classifier.fit(tr_x,tr_y)


# In[38]:


# KFold Cross Validation

Kfolds = StratifiedKFold(n_splits=5,shuffle = True , random_state = 21)
i = 1
roc_auc = []
predicted = []

for train_index,val_index in Kfolds.split(train_x,train_y):
    tr_x , val_x = train_x.iloc[train_index], train_x.iloc[val_index]
    tr_y , val_y = train_y.iloc[train_index], train_y.iloc[val_index]
    
    lgb = LGBMClassifier(learning_rate=0.01, max_bin=50, max_depth=14, n_estimators=600, num_leaves=37, subsample=0.75)
    lgb.fit(tr_x , tr_y)
    valid_pred = lgb.predict(val_x) ; train_pred = lgb.predict(tr_x)
    prob_val = lgb.predict_proba(val_x)[:,1] ; prob_train = lgb.predict_proba(tr_x)[:,1]
    valid_score = roc_auc_score(val_y, prob_val) ; train_score = roc_auc_score(tr_y,prob_train)
    
    predicted.append(list(lgb.predict(test_new)))
    
    print("For CV = ",i)
    print("\nROCAUC for Validation:",valid_score,'||',"\tROCAUC for Train:",train_score)    
#    print("\nClassification Report\n",classification_report(y_pred,val_y))
    print("\n***************")
    i = i+1
    roc_auc.append(valid_score)
    
print("Mean Accuracy",np.mean(roc_auc))


# In[39]:


# a = pd.DataFrame({1:predicted[0],2:predicted[1],3:predicted[2],4:predicted[3],  5:predicted[4]})
# final_pred = a.mode(axis = 1)


# # Submission



# In[40]:


train_x.columns


# In[41]:


test_new.columns



# Save model to disk
import pickle
pickle.dump(lgb, open('model.pkl','wb'))

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict())
# In[43]:


sub = pd.read_csv(r"E:\Data science\DataHack\Analytics Vidhya\Job_a_thon_28052021\sample_submission_eyYijxG.csv")
sub['Is_Lead'] = final_pred


# In[44]:


sub.head()


# In[45]:


sub.to_csv('HerokuDeployment/Credit_Card_Leads.csv', index=False)
a


train_x.head()
full_df.head()



