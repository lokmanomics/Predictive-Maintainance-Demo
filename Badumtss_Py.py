#!/usr/bin/env python
# coding: utf-8

# Author: Ahmad Lokman Anuar

# __Introduction__
# 
# APS (Air Pressure System) Failure and Operational Data for Scania Trucks. Download dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/00421/.
# 
# 

# __Approach__
# 
# General work flow: 
# 
# 0. Manual review of the data.
# 1. Environment setup and load dataset.
# 2. Descriptive Analysis to summarize data <br>
# 3. Data Exploration and PreProcessing
# 4. Apply ML classifiers 
# 4.5. Observe performance
# 5. Performance tuning
# 6. Finalize model & results.
# 

# ## __0. Manual Review Dataset__
# 
# It is beneficial to skim through dataset manually for obvious pointers that can be rectified immediately. 
# 
# Upon inspection, the dataset is in CSV format and contains non-tabulated header paragraph that will interfere with downstream processing. 
# 
# Hence, the header paragraph is manually deleted.

# ## __1. Set up environment__

# In[1]:


import numpy as np
import pandas as pd
import sklearn as skl

from datetime import datetime
from matplotlib import pyplot

from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# First, we load CSV (directory sensitive) to dataframe.

# In[2]:


startTimeScript = datetime.now()

file_path_train = 'aps_failure_training_set.csv' #directory of training dataset
df_train = pd.read_csv(file_path_train, na_values="na") #write csv into dataframe 

file_path_test = 'aps_failure_test_set.csv'
df_test = pd.read_csv(file_path_test, na_values="na")


# View top 5 row in training dataset

# In[ ]:


df_train.head() #view top 5 of training dataset 


# View top 5 row in test dataset.

# In[ ]:


df_test.head() #view top 5 of training dataset 


# ## __2. Descriptive Analysis.__
# 
# Before delving deeper, we apply several Descriptive Analysis to know the state and condition of our dataset.

# ### __2.1 Size of dataset.__

# In[3]:


print("----- Size of Dataset -----")
print(f"Train : {df_train.shape} | Test : {df_test.shape}")


# ### __2.2 General Description of Dataset.__

# In[4]:


print("----- Dataset Statistics -----")
print(df_train.describe())


# ### __2.3 Class Distribution.__

# In[5]:


print("----- Class Distibution -----")
print("Number of positive classes = ", sum(df_train['class'] == 'pos'))
print("Number of negative classes = ", sum(df_train['class'] == 'neg'))


# ### __2.4 Any Missing Values?__

# In[6]:


print("----- Train Data: Missing Values Count by Attributes -----")
print(df_train.isnull().sum())


# In[7]:


print("----- Test Data: Missing Values Count by Attributes -----")
print(df_test.isnull().sum())


# ## __3. Preprocessing.__
# 
# With the knowledge about the dataset conditions obtained from exploratory analysis in prior phase, we will process dataset accordingly to manage features and mitigate unfavorable factors such as outliers, missing values and imbalance that can affect downstream works.

# ### __3.1 Feature Engineering.__
# 
# From the dataset description, it is understood that there would be only two classes ("neg" and "pos"), so we will treat it as binary classification problem and substitute class -> neg to 0 and pos to 1.

# In[8]:


# Replace class labels with integer values (neg = 0, pos = 1) in training and test data-set
df_train['class'].replace({
    'neg': 0,
    'pos': 1
}, inplace=True)
df_test['class'].replace({
    'neg': 0,
    'pos': 1
}, inplace=True)


# We observe the 'class' values that has been changed to 0 and 1 as follows.

# In[9]:


df_train.head()


# ### __3.2 Manage Missing Values.__

# We impute all missing values that are present with mean values from each columns. 
# 
# Several other strategies can be applied, such as removal of columns with NAs that exceed threshold composition, or impute the NAs with other values (0,1,median,etc).

# In[14]:


# Fill missing values in training and test dataset
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)


# In[15]:


#Check dataset size
print(f"Train : {df_train.shape} | Test : {df_test.shape}")


# We verify that no missing values still present in the dataset

# In[16]:


print(df_train.isnull().sum())


# ### __3.3 Address Unbalanced Data.__

# Recall unbalanced data class of negative and positive:

# In[17]:


pyplot.close('all')
bins = np.bincount(df_train['class'].values)
pyplot.bar([0,1], bins, color='maroon')
pyplot.xticks([0,1])
pyplot.xlabel('Classes')
pyplot.ylabel('Count')
pyplot.title('Histogram of target classes [train set]')
pyplot.show()


# This is an imbalance class where the positive (1) target variable is much lesser than the negative (0). However, we can treat this imbalance dataset using SMOTE (Synthetic Minorty OverSampling Technique), a method to oversample minority class in out dataset.
# 
# Other options to balance class ratio is to undersample majority class, using different sampling algorithms.

# In[96]:


# Extract features and labels from the training and test data-set
y_train = df_train.loc[:, 'class']
x_train = df_train.drop('class', axis=1)
x_test = df_test.drop('class', axis=1)
y_test = df_test.loc[:, 'class']

print("X_train.shape: {} Y_train.shape: {}".format(x_train.shape, y_train.shape))
print("X_validation.shape: {} Y_validation.shape: {}".format(x_test.shape, y_test.shape))


# In[105]:


#  Synthetic Minority Oversampling Technique to balance the training data-set
sm = SMOTE()
x_train, y_train = sm.fit_sample(x_train, y_train)


# Let us check oversampling effect to our dataset size:

# In[115]:


print(f"Train : {y_train.shape} | Test : {y_test.shape}")


# To check number of classes after oversampling,

# In[111]:


y_train_smoted = y_train.to_frame()
y_train_smoted["class"].value_counts()


# Or as in histogram:

# In[116]:


pyplot.close('all')
bins = np.bincount(y_train_smoted['class'].values)
pyplot.bar([0,1], bins, color='maroon')
pyplot.xticks([0,1])
pyplot.xlabel('Classes')
pyplot.ylabel('Count')
pyplot.title('Histogram of target classes [train set]')
pyplot.show()


# The dataset has been considerably inflated to equal amount of positive and negative cases (59000 each).
# 
# Here we have completed Pre-Processing of our dataset:
# - No missing values, all has been imputated with Mean.
# - No imbalanced data class, Oversampling of minority class balanced the composition.
# 
# The data is fit to be implemented with ML classifiers.

# ## 4. Apply ML Classifiers

# Set test options and evaluation metric

# In[72]:


# Run algorithms using 10-fold cross validation
num_folds = 10
scoring = 'accuracy'


# In[75]:


from sklearn.linear_model import LogisticRegression
seedNum = 1234 #seed number to allow reproducibility

# Set up Algorithms package to test 
models = []

#Algorithms to be tested here are LG,GaussianNB,Random Forest, and GB using default parameters
models.append(('LR', LogisticRegression(random_state=seedNum)))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(random_state=seedNum)))
models.append(('GBM', GradientBoostingClassifier(random_state=seedNum)))

results = [] #init
names = [] #init
metrics = [] #init


# In[77]:


#Generate models stat in order
for name, model in models:
    startTimeModule = datetime.now()
    #kfold used as method to compute weights across iterations
    kfold = KFold(n_splits=num_folds, random_state=seedNum, shuffle=True)
    #crossvalidate kfold and score 
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    metrics.append(cv_results.mean())
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print ('Model training time:',(datetime.now() - startTimeModule))
print ('Average metrics ('+scoring+') from all models:',np.mean(metrics))


# ### 4.5 Compare performance of algorithms

# In[78]:


fig = pyplot.figure()
fig.suptitle('Baseline Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Random Forrest classifier outperforms other algorithm, reporting 99.67% accuracy. Hence RF will be used to predictive analysis with the test data.

# ### 5. Improving accuracy

# Hyperparameter tuning to find best parameters for Random Forrest classifiers. For this demonstration, we will perform tuning to find n Decision Tree branch that yields the best performance. 
# 
# We set the n_estimator parameter for RF to compute n of 75, 100, 125, 150, 175.

# In[80]:


# Initialize RF result array
results = []
names = []

# Tuning algorithms  
startTimeModule = datetime.now()
#Here we try to find best n_estimator, we can define other RF params in dict to invoke tuning as well
paramGrid1 = dict(n_estimators=np.array([75,100,125,150,175])) #num of decision trees to try
model1 = RandomForestClassifier(random_state=seedNum)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seedNum)
#compound  cross validate values for hyperparameter tuning
grid1 = GridSearchCV(estimator=model1, param_grid=paramGrid1, scoring=scoring, cv=kfold) 
grid_result1 = grid1.fit(x_train, y_train) #actualparamtraining

print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))
results.append(grid_result1.cv_results_['mean_test_score'])
names.append('RF')
means = grid_result1.cv_results_['mean_test_score']
stds = grid_result1.cv_results_['std_test_score']
params = grid_result1.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print ('Model training time:',(datetime.now() - startTimeModule))


# For the hyperparameter tuning results, we found that value of  n_estimators parameter that returns highest accuracy is 150. Hence we will use this knowledge to tune our Random Forrest algorithm to run with test data.

# ### 6. Finalize model
# 
# We will retrain the classifier using our training dataset with best parameter ('n_estimators'=150) to build an optimal model. This optimum model will be used to predict from test dataset.
# 
# Creating standalone model using the tuned parameters
# Saving an optimal model to file or blob for later use.

# 6.1. Prediction for test set

# In[121]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Use best parameter along with the model of choice to the test dataset
model = RandomForestClassifier(n_estimators=150, random_state=seedNum)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# 6.2. Create standalone model on entire training dataset

# In[122]:


#Training RF model with best params found to finalize
startTimeModule = datetime.now()
finalModel = RandomForestClassifier(n_estimators=150, random_state=seedNum)
finalModel.fit(x_train, y_train)
print ('Model training time:',(datetime.now() - startTimeModule))


# 6.3 Save model for later use

# In[124]:


#modelName = 'finalModel_BinaryClass.sav'
#dump(finalModel, modelName)

print ('Total time for the script:',(datetime.now() - startTimeScript))


# Challenge Metric

# In[126]:


cm = confusion_matrix(y_test, predictions).ravel()
cm = pd.DataFrame(cm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])

total_cost = 10*(cm.FP[0]) + 500*(cm.FN[0])
print(f"Total cost for RF model is {total_cost}")


# In[ ]:




