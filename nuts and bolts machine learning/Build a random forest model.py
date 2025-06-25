#!/usr/bin/env python
# coding: utf-8

# # Exemplar: Build a random forest model

# ## **Introduction**
# 
# 
# As you're learning, random forests are popular statistical learning algorithms. Some of their primary benefits include reducing variance, bias, and the chance of overfitting.
# 
# This activity is a continuation of the project you began modeling with decision trees for an airline. Here, you will train, tune, and evaluate a random forest model using data from spreadsheet of survey responses from 129,880 customers. It includes data points such as class, flight distance, and inflight entertainment. Your random forest model will be used to predict whether a customer will be satisfied with their flight experience.
# 
# **Note:** Because this lab uses a real dataset, this notebook first requires exploratory data analysis, data cleaning, and other manipulations to prepare it for modeling.

# ## **Step 1: Imports** 

# Import relevant Python libraries and modules, including `numpy` and `pandas`libraries for data processing; the `pickle` package to save the model; and the `sklearn` library, containing:
# - The module `ensemble`, which has the function `RandomForestClassifier`
# - The module `model_selection`, which has the functions `train_test_split`, `PredefinedSplit`, and `GridSearchCV` 
# - The module `metrics`, which has the functions `f1_score`, `precision_score`, `recall_score`, and `accuracy_score`

# In[1]:


# Import `numpy`, `pandas`, `pickle`, and `sklearn`.
# Import the relevant functions from `sklearn.ensemble`, `sklearn.model_selection`, and `sklearn.metrics`.

### YOUR CODE HERE ###
 
import numpy as np
import pandas as pd

import pickle as pkl
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

### YOUR CODE HERE ###

air_data = pd.read_csv("Invistico_Airline.csv")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `read_csv()` function from the `pandas` library can be helpful here.
#  
# </details>

# Now, you're ready to begin cleaning your data. 

# ## **Step 2: Data cleaning** 

# To get a sense of the data, display the first 10 rows.

# In[3]:


# Display first 10 rows.

### YOUR CODE HERE ###

air_data.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `head()` function from the `pandas` library can be helpful here.
#  
# </details>

# Now, display the variable names and their data types. 

# In[4]:


# Display variable names and types.

### YOUR CODE HERE ###

air_data.dtypes


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# DataFrames have an attribute that outputs variable names and data types in one result.
#  
# </details>

# **Question:** What do you observe about the differences in data types among the variables included in the data?
# 
# There are three types of variables included in the data: int64, float64, and object. The object variables are satisfaction, customer type, type of travel, and class. 

# Next, to understand the size of the dataset, identify the number of rows and the number of columns.

# In[5]:


# Identify the number of rows and the number of columns.

### YOUR CODE HERE ###

air_data.shape


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a method in the `pandas` library that outputs the number of rows and the number of columns in one result.
# 
# </details>

# Now, check for missing values in the rows of the data. Start with .isna() to get Booleans indicating whether each value in the data is missing. Then, use .any(axis=1) to get Booleans indicating whether there are any missing values along the columns in each row. Finally, use .sum() to get the number of rows that contain missing values.

# In[6]:


# Get Booleans to find missing values in data.
# Get Booleans to find missing values along columns.
# Get the number of rows that contain missing values.

### YOUR CODE HERE ###

air_data.isna().any(axis=1).sum()


# **Question:** How many rows of data are missing values?**
# 
# There are 393 rows with missing values.

# Drop the rows with missing values. This is an important step in data cleaning, as it makes the data more useful for analysis and regression. Then, save the resulting pandas DataFrame in a variable named `air_data_subset`.

# In[7]:


# Drop missing values.
# Save the DataFrame in variable `air_data_subset`.

### YOUR CODE HERE ###

air_data_subset = air_data.dropna(axis=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `dropna()` function is helpful here.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The axis parameter passed in to this function should be set to 0 (if you want to drop rows containing missing values) or 1 (if you want to drop columns containing missing values).
# </details>

# Next, display the first 10 rows to examine the data subset.

# In[8]:


# Display the first 10 rows.

### YOUR CODE HERE ###

air_data_subset.head(10)


# Confirm that it does not contain any missing values.

# In[9]:


# Count of missing values.

### YOUR CODE HERE ###

air_data_subset.isna().sum()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can use the `.isna().sum()` to get the number of missing values for each variable.
# 
# </details>

# Next, convert the categorical features to indicator (one-hot encoded) features. 
# 
# **Note:** The `drop_first` argument can be kept as default (`False`) during one-hot encoding for random forest models, so it does not need to be specified. Also, the target variable, `satisfaction`, does not need to be encoded and will be extracted in a later step.

# In[10]:


# Convert categorical features to one-hot encoded features.

### YOUR CODE HERE ###

air_data_subset_dummies = pd.get_dummies(air_data_subset, 
                                         columns=['Customer Type','Type of Travel','Class'])


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can use the `pd.get_dummies()` function to convert categorical variables to one-hot encoded variables.
# </details>

# **Question:** Why is it necessary to convert categorical data into dummy variables?**
# 
# It is necessary because the sklearn implementation of `RandomForestClassifier()` requires that categorical features be encoded to numeric, which can be done using dummy variables or one-hot encoding.

# Next, display the first 10 rows to review the `air_data_subset_dummies`. 

# In[11]:


# Display the first 10 rows.

### YOUR CODE HERE ###

air_data_subset_dummies.head(10)


# Then, check the variables of air_data_subset_dummies.

# In[12]:


# Display variables.

### YOUR CODE HERE ###

air_data_subset_dummies.dtypes


# **Question:** What changes do you observe after converting the string data to dummy variables?**
# 
# All of the following changes could be observed: 
# 
# - Customer Type  -->  Customer Type_Loyal Customer and Customer Type_disloyal Customer
# - Type of Travel -->  Type of Travel_Business travel and Type of Travel_Personal travel 
# - Class          --> Class_Business, Class_Eco, Class_Eco Plus

# ## **Step 3: Model building** 

# The first step to building your model is separating the labels (y) from the features (X).

# In[13]:


# Separate the dataset into labels (y) and features (X).

### YOUR CODE HERE ###

y = air_data_subset_dummies["satisfaction"]
X = air_data_subset_dummies.drop("satisfaction", axis=1)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Save the labels (the values in the `satisfaction` column) as `y`.
# 
# Save the features as `X`. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# To obtain the features, drop the `satisfaction` column from the DataFrame.
# 
# </details>

# Once separated, split the data into train, validate, and test sets. 

# In[14]:


# Separate into train, validate, test sets.

### YOUR CODE HERE ###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `train_test_split()` function twice to create train/validate/test sets, passing in `random_state` for reproducible results. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Split `X`, `y` to get `X_train`, `X_test`, `y_train`, `y_test`. Set the `test_size` argument to the proportion of data points you want to select for testing. 
# 
# Split `X_train`, `y_train` to get `X_tr`, `X_val`, `y_tr`, `y_val`. Set the `test_size` argument to the proportion of data points you want to select for validation. 
# 
# </details>

# ### Tune the model
# 
# Now, fit and tune a random forest model with separate validation set. Begin by determining a set of hyperparameters for tuning the model using GridSearchCV.

# In[15]:


# Determine set of hyperparameters.

### YOUR CODE HERE ###

cv_params = {'n_estimators' : [50,100], 
              'max_depth' : [10,50],        
              'min_samples_leaf' : [0.5,1], 
              'min_samples_split' : [0.001, 0.01],
              'max_features' : ["sqrt"], 
              'max_samples' : [.5,.9]}


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Create a dictionary `cv_params` that maps each hyperparameter name to a list of values. The GridSearch you conduct will set the hyperparameter to each possible value, as specified, and determine which value is optimal.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The main hyperparameters here include `'n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features', and 'max_samples'`. These will be the keys in the dictionary `cv_params`.
# 
# </details>

# Next, create a list of split indices.

# In[16]:


# Create list of split indices.

### YOUR CODE HERE ###

split_index = [0 if x in X_val.index else -1 for x in X_train.index]
custom_split = PredefinedSplit(split_index)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use list comprehension, iterating over the indices of `X_train`. The list can consists of 0s to indicate data points that should be treated as validation data and -1s to indicate data points that should be treated as training data.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `PredfinedSplit()`, passing in `split_index`, saving the output as `custom_split`. This will serve as a custom split that will identify which data points from the train set should be treated as validation data during GridSearch.
# 
# </details>

# Now, instantiate your model.

# In[17]:


# Instantiate model.

### YOUR CODE HERE ### 

rf = RandomForestClassifier(random_state=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `RandomForestClassifier()`, specifying the `random_state` argument for reproducible results. This will help you instantiate a random forest model, `rf`.
# 
# </details>

# Next, use GridSearchCV to search over the specified parameters.

# In[18]:


# Search over specified parameters.

### YOUR CODE HERE ### 

rf_val = GridSearchCV(rf, cv_params, cv=custom_split, refit='f1', n_jobs = -1, verbose = 1)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `GridSearchCV()`, passing in `rf` and `cv_params` and specifying `cv` as `custom_split`. Additional arguments that you can specify include: `refit='f1', n_jobs = -1, verbose = 1`. 
# 
# </details>

# Now, fit your model.

# In[19]:


get_ipython().run_cell_magic('time', '', '\n# Fit the model.\n\n### YOUR CODE HERE ###\n\n\nrf_val.fit(X_train, y_train)')


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `fit()` method to train the GridSearchCV model on `X_train` and `y_train`. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Add the magic function `%%time` to keep track of the amount of time it takes to fit the model and display this information once execution has completed. Remember that this code must be the first line in the cell.
# 
# </details>

# Finally, obtain the optimal parameters.

# In[20]:


# Obtain optimal parameters.

### YOUR CODE HERE ###

rf_val.best_params_


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `best_params_` attribute to obtain the optimal values for the hyperparameters from the GridSearchCV model.
# 
# </details>

# ## **Step 4: Results and evaluation** 

# Use the selected model to predict on your test data. Use the optimal parameters found via GridSearchCV.

# In[21]:


# Use optimal parameters on GridSearchCV.

### YOUR CODE HERE ###

rf_opt = RandomForestClassifier(n_estimators = 50, max_depth = 50, 
                                min_samples_leaf = 1, min_samples_split = 0.001,
                                max_features="sqrt", max_samples = 0.9, random_state = 0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `RandomForestClassifier()`, specifying the `random_state` argument for reproducible results and passing in the optimal hyperparameters found in the previous step. To distinguish this from the previous random forest model, consider naming this variable `rf_opt`.
# 
# </details>

# Once again, fit the optimal model.

# In[22]:


# Fit the optimal model.

### YOUR CODE HERE ###

rf_opt.fit(X_train, y_train)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `fit()` method to train `rf_opt` on `X_train` and `y_train`.
# 
# </details>

# And predict on the test set using the optimal model.

# In[23]:


# Predict on test set.

### YOUR CODE HERE ###

y_pred = rf_opt.predict(X_test)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `predict()` function to make predictions on `X_test` using `rf_opt`. Save the predictions now (for example, as `y_pred`), to use them later for comparing to the true labels. 
# 
# </details>

# ### Obtain performance scores

# First, get your precision score.

# In[24]:


# Get precision score.

### YOUR CODE HERE ###

pc_test = precision_score(y_test, y_pred, pos_label = "satisfied")
print("The precision score is {pc:.3f}".format(pc = pc_test))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `precision_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# Then, collect the recall score.

# In[25]:


# Get recall score.

### YOUR CODE HERE ###

rc_test = recall_score(y_test, y_pred, pos_label = "satisfied")
print("The recall score is {rc:.3f}".format(rc = rc_test))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `recall_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# Next, obtain your accuracy score.

# In[26]:


# Get accuracy score.

### YOUR CODE HERE ###

ac_test = accuracy_score(y_test, y_pred)
print("The accuracy score is {ac:.3f}".format(ac = ac_test))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `accuracy_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# Finally, collect your F1-score.

# In[27]:


# Get F1 score.

### YOUR CODE HERE ###

f1_test = f1_score(y_test, y_pred, pos_label = "satisfied")
print("The F1 score is {f1:.3f}".format(f1 = f1_test))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# You can call the `f1_score()` function from `sklearn.metrics`, passing in `y_test` and `y_pred` and specifying the `pos_label` argument as `"satisfied"`.
# </details>

# **Question:** How is the F1-score calculated?
# 
# F1 scores are calculated using the following formula: 
# 
# F1 = 2 * (precision * recall) / (precision + recall)

# **Question:** What are the pros and cons of performing the model selection using test data instead of a separate validation dataset?
# 
# Pros: <br />
# *  The coding workload is reduced.
# *  The scripts for data splitting are shorter.
# *  It's only  necessary to evaluate test dataset performance once, instead of two evaluations (validate and test).
# 
# Cons: <br />
# * If a model is evaluated using samples that were also used to build or fine-tune that model, it likely will provide a biased evaluation.
# * A potential overfitting issue could happen when fitting the model's scores on the test data.
# 
# 

# ### Evaluate the model
# 
# Now that you have results, evaluate the model. 

# **Question:** What are the four basic parameters for evaluating the performance of a classification model?
# 
# 1. True positives (TP): These are correctly predicted positive values, which means the value of actual and predicted classes are positive. 
# 
# 2. True negatives (TN): These are correctly predicted negative values, which means the value of the actual and predicted classes are negative.
# 
# 3. False positives (FP): This occurs when the value of the actual class is negative and the value of the predicted class is positive.
# 
# 4. False negatives (FN): This occurs when the value of the actual class is positive and the value of the predicted class in negative. 
# 
# **Reminder:** When fitting and tuning classification modeld, data professioals aim to minimize false positives and false negatives.

# **Question:**  What do the four scores demonstrate about your model, and how do you calculate them?
# 
# - Accuracy (TP+TN/TP+FP+FN+TN): The ratio of correctly predicted observations to total observations. 
#  
# - Precision (TP/TP+FP): The ratio of correctly predicted positive observations to total predicted positive observations. 
# 
# - Recall (Sensitivity, TP/TP+FN): The ratio of correctly predicted positive observations to all observations in actual class.
# 
# - F1 score: The harmonic average of precision and recall, which takes into account both false positives and false negatives. 

# Calculate the scores: precision score, recall score, accuracy score, F1 score.

# In[28]:


# Precision score on test data set.

### YOUR CODE HERE ###

print("\nThe precision score is: {pc:.3f}".format(pc = pc_test), "for the test set,", "\nwhich means of all positive predictions,", "{pc_pct:.1f}% prediction are true positive.".format(pc_pct = pc_test * 100))


# In[29]:


# Recall score on test data set.

### YOUR CODE HERE ###

print("\nThe recall score is: {rc:.3f}".format(rc = rc_test), "for the test set,", "\nwhich means of which means of all real positive cases in test set,", "{rc_pct:.1f}% are  predicted positive.".format(rc_pct = rc_test * 100))


# In[30]:


# Accuracy score on test data set.

### YOUR CODE HERE ###

print("\nThe accuracy score is: {ac:.3f}".format(ac = ac_test), "for the test set,", "\nwhich means of all cases in test set,", "{ac_pct:.1f}% are predicted true positive or true negative.".format(ac_pct = ac_test * 100))


# In[31]:


# F1 score on test data set.

### YOUR CODE HERE ###

print("\nThe F1 score is: {f1:.3f}".format(f1 = f1_test), "for the test set,", "\nwhich means the test set's harmonic mean is {f1_pct:.1f}%.".format(f1_pct = f1_test * 100))


# **Question:** How does this model perform based on the four scores?
# 
# The model performs well according to all 4 performance metrics. The model's precision score is slightly better than the 3 other metrics. 

# ### Evaluate the model
# 
# Finally, create a table of results that you can use to evaluate the performace of your model.

# In[32]:


# Create table of results.

### YOUR CODE HERE ###
table = pd.DataFrame({'Model': ["Tuned Decision Tree", "Tuned Random Forest"],
                        'F1':  [0.945422, f1_test],
                        'Recall': [0.935863, rc_test],
                        'Precision': [0.955197, pc_test],
                        'Accuracy': [0.940864, ac_test]
                      }
                    )
table


# 
# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Build a table to compare the performance of the models. Create a DataFrame using the `pd.DataFrame()` function.
# 
# </details>

# **Question:** How does the random forest model compare to the decision tree model you built in the previous lab?
# 
# The tuned random forest has higher scores overall, so it is the better model. Particularly, it shows a better F1 score than the decision tree model, which indicates that the random forest model may do better at classification when taking into account false positives and false negatives. 
# 
# 

# ## **Considerations**
# 
# 
# **What are the key takeaways from this lab?**
# - Data exploring, cleaning, and encoding are necessary for model building.
# - A separate validation set is typically used for tuning a model, rather than using the test set. This also helps avoid the evaluation becoming biased.
# -  F1 scores are usually more useful than accuracy scores. If the cost of false positives and false negatives are very different, it’s better to use the F1 score and combine the information from precision and recall. 
# * The random forest model yields a more effective performance than a decision tree model. 
# 
# **What summary would you provide to stakeholders?**
# * The random forest model predicted satisfaction with more than 94.2% accuracy. The precision is over 95% and the recall is approximately 94.5%. 
# * The random forest model outperformed the tuned decision tree with the best hyperparameters in most of the four scores. This indicates that the random forest model may perform better.
# * Because stakeholders were interested in learning about the factors that are most important to customer satisfaction, this would be shared based on the tuned random forest. 
# * In addition, you would provide details about the precision, recall, accuracy, and F1 scores to support your findings. 

# ### References

# [What is the Difference Between Test and Validation Datasets?,  Jason Brownlee](https://machinelearningmastery.com/difference-test-validation-datasets/)
# 
# [Decision Trees and Random Forests Neil Liberman](https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
