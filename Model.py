import pandas as pd 
import numpy as np
import os
import re
import warnings

#ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style

#%matplotlib inline

import sklearn 
import math

#relevant ML libraries
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn.metrics as metrics
from sklearn.neighbors import LocalOutlierFactor

#ML models
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier

# saving model
import pickle

warnings.filterwarnings('ignore')


LP = pd.read_csv('Loans data.csv')

LPcat = LP.describe(include = 'object')

for col in LPcat.columns:
    LPcat = col

LPcat =  LP[['Loan ID', 
             'Customer ID', 
             'Loan Status', 
             'Term', 
             'Years in current job', 
             'Home Ownership', 
             'Purpose', 
             'Maximum Open Credit']]

LPcat.drop('Loan ID', axis = 1, inplace = True)
LPcat.drop('Customer ID', axis = 1, inplace = True)


LPcat['Maximum Open Credit'].replace('#VALUE!', np.nan, inplace = True)
LPcat['Maximum Open Credit'][4930]

LPcat['Maximum Open Credit'] = pd.to_numeric(LPcat['Maximum Open Credit'])
LP['Maximum Open Credit'] = LPcat['Maximum Open Credit']


LPcat.drop('Maximum Open Credit', axis = 1, inplace = True)

LPcat['Years in current job']=LPcat['Years in current job'].str.replace(
    '<','',).str.replace('+','').str.replace('year','').str.replace('s','').astype('float')

LP['Years in current job']=LPcat['Years in current job']
LPcat.drop('Years in current job', axis = 1, inplace = True)


term_dummies = pd.get_dummies(LPcat['Term'])
LPcat = pd.concat([LPcat,term_dummies],axis = 1)
LPcat.drop('Term', axis = 1, inplace = True)


LPcat['Home Ownership'].replace(['HaveMortgage', 'Own Home', 'Rent', 'Home Mortgage'], [0, 1, 2, 3], inplace = True) 


LPcat['Purpose'].replace(['Debt Consolidation',
                          'other', 
                          'Home Improvements', 
                          'Other', 
                          'Business Loan', 
                          'Buy a Car', 
                          'Medical Bills', 
                          'Buy House',     
                          'Take a Trip', 
                          'major_purchase', 
                          'small_business', 
                          'moving', 
                          'wedding', 
                          'Educational Expenses', 
                          'vacation', 
                          'renewable_energy'], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], inplace = True) 

LPcat['Loan Status'].replace(['Loan Refused','Loan Given'],[0,1], inplace =True)

LPnum = LP.describe(include = np.number)
LPnum = LP[[ 'Current Loan Amount',
             'Credit Score',
             'Years in current job',
             'Annual Income', 
             'Monthly Debt', 
             'Years of Credit History',
             'Months since last delinquent',
             'Number of Open Accounts',
             'Number of Credit Problems',
             'Current Credit Balance',
             'Bankruptcies',
             'Maximum Open Credit',
            'Tax Liens' ]]

def LP_column_drop(LPnum, feature_name):      
    LPnum.drop(feature_name, axis = 1, inplace = True)
    
    
LP_column_drop(LPnum, 'Bankruptcies')
LP_column_drop(LPnum, 'Tax Liens')            

def dist(LPnum,feature_name):
    sns.distplot(LPnum[feature_name])
    plt.show()
def checker(LPnum,feature_name):
    sns.boxplot(LPnum[feature_name],orient = 'v')
    plt.show()
def remover(LPnum,feature_name):
    mean_ = LPnum[feature_name].mean()
    std_ = LPnum[feature_name].std()
    cut_off =std_*3
    lower,upper = mean_-std_ , mean_+std_
    checker = (LPnum[feature_name]<upper) & (LPnum[feature_name]>lower)
    std_method = LPnum[feature_name][checker]
    LPnum[feature_name]=std_method
def outlier_remover(LPnum,feature_name):
    dist(LPnum,feature_name)
    checker(LPnum,feature_name)
    remover(LPnum,feature_name)
    
    

outlier_remover(LPnum, 'Current Loan Amount')
outlier_remover(LPnum, 'Credit Score')
outlier_remover(LPnum, 'Annual Income')
outlier_remover(LPnum, 'Monthly Debt')
outlier_remover(LPnum, 'Years of Credit History')
outlier_remover(LPnum, 'Number of Open Accounts')
outlier_remover(LPnum, 'Number of Credit Problems')
outlier_remover(LPnum, 'Current Credit Balance')
outlier_remover(LPnum, 'Maximum Open Credit')


LPnum_mask = LPnum.isnull().sum()/len(LPnum)<.3
LPnum = LPnum.loc[:,LPnum_mask]

LPnum['Current Loan Amount'].fillna(LPnum['Current Loan Amount'].mean(), inplace = True)
LPnum['Years in current job'].fillna(LPnum['Years in current job'].mean(), inplace = True)
LPnum['Annual Income'].fillna(LPnum['Annual Income'].mean(), inplace = True)
LPnum['Credit Score'].fillna(LPnum['Credit Score'].mean(), inplace = True)
LPnum['Monthly Debt'].fillna(LPnum['Monthly Debt'].mean(), inplace = True)
LPnum['Years of Credit History'].fillna(LPnum['Years of Credit History'].mean(), inplace = True)
LPnum['Number of Open Accounts'].fillna(LPnum['Number of Open Accounts'].mean(), inplace = True)
LPnum['Number of Credit Problems'].fillna(LPnum['Number of Credit Problems'].mean(), inplace = True)
LPnum['Current Credit Balance'].fillna(LPnum['Current Credit Balance'].mean(), inplace = True)
LPnum['Maximum Open Credit'].fillna(LPnum['Maximum Open Credit'].mean(), inplace = True)

LPF = pd.concat([LPcat, LPnum], axis = 1, join = 'inner')

LP_column_drop(LPF, 'Number of Credit Problems')

y= LPF['Loan Status']                         # Target Variable
X = LPF.drop('Loan Status', axis =1)          # Independent Variable

scaler = StandardScaler()
LPFs = scaler.fit_transform(X)
LPFn = pd.DataFrame(LPFs, columns = X.columns)

Loan_Given = LPF[LPF['Loan Status']==1]
Loan_Refused = LPF[LPF['Loan Status']==0]

X_resampled,y_resampled = ADASYN().fit_resample(X,y)

X = X_resampled
y = y_resampled

X.drop('Purpose', axis = 1, inplace = True)
X.drop('Monthly Debt', axis = 1, inplace = True)
X.drop('Current Credit Balance', axis = 1, inplace = True)
X.drop('Maximum Open Credit', axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#from sklearn.ensemble import 
model = HistGradientBoostingClassifier()
learning_rate = [0.001, 0.01, 0.1]
max_depth = [3, 7, 9]

# define grid search
grid = dict(learning_rate=learning_rate, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

LPhgb = HistGradientBoostingClassifier(**grid_result.best_params_, max_bins=255, max_iter=100)

#Fit 'LPdt' to the training set
LPhgb.fit(X_train, y_train)

# Predict Output
dy_predict = LPhgb.predict(X_test)
pred_LPhgb = LPhgb.predict_proba(X_test)[:,1]

#Train and Test Scores
LPhgb_Tr_Score = round(LPhgb.score(X_train, y_train)*100, 2)
LPhgb_Tt_Score = round(LPhgb.score(X_test, y_test)*100, 2)
print('train set score: {:.2f}'.format(LPhgb_Tr_Score))
print('test set score: {:.2f}'.format(LPhgb_Tt_Score))

pickle.dump(LPhgb, open('ML_LoanPrediction_Model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('ML_LoanPrediction_Model.pkl','rb'))
lp = model.predict([[2,1,0,1,12232.0,716.612735,1.0,46643.0,777.39,18.0,12.0,6762,7946]])
if lp == 0:
    LPred = print(f'Sorry, you do not qualify for a loan at this time.')
elif lp == 1:
    LPred = print(f'Congralutaions, you qualify for a loan.')
else:
    LPred = print('please speak with a customser care repersentative.')
