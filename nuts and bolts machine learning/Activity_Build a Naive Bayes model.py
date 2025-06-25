#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a Naive Bayes model 

# ## Introduction
# 
# In this activity, you will build your own Naive Bayes model. Naive Bayes models can be valuable to use any time you are doing work with predictions because they give you a way to account for new information. In today's world, where data is constantly evolving, modeling with Naive Bayes can help you adapt quickly and make more accurate predictions about what could occur.
# 
# For this activity, you work for a firm that provides insights for management and coaches in the National Basketball Association (NBA), a professional basketball league in North America. The league is interested in retaining players who can last in the high-pressure environment of professional basketball and help the team be successful over time. In the previous activity, you analyzed a subset of data that contained information about the NBA players and their performance records. You conducted feature engineering to determine which features would most effectively predict a player's career duration. You will now use those insights to build a model that predicts whether a player will have an NBA career lasting five years or more. 
# 
# The data for this activity consists of performance statistics from each player's rookie year. There are 1,341 observations, and each observation in the data represents a different player in the NBA. Your target variable is a Boolean value that indicates whether a given player will last in the league for five years. Since you previously performed feature engineering on this data, it is now ready for modeling.   

# ## Step 1: Imports

# ### Import packages
# 
# Begin with your import statements. Of particular note here are `pandas` and from `sklearn`, `naive_bayes`, `model_selection`, and `metrics`.

# In[1]:


# Import relevant libraries and modules.
import pandas as pd
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics


# ### Load the dataset
# 
# Recall that in the lab about feature engineering, you outputted features for the NBA player dataset along with the target variable ``target_5yrs``. Data was imported as a DataFrame called `extracted_data`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[4]:


# RUN THIS CELL TO IMPORT YOUR DATA.
# Load extracted_nba_players_data.csv into a DataFrame called extracted_data.

extracted_data = pd.read_csv('extracted_nba_players_data.csv')


# ### Display the data
# 
# Review the first 10 rows of data.

# In[5]:


# Display the first 10 rows of data.



extracted_data.head(10)


# ## Step 2: Model preparation

# ### Isolate your target and predictor variables
# Separately define the target variable (`target_5yrs`) and the features.

# In[6]:


# Define the y (target) variable.

y = extracted_data['target_5yrs']


# Define the X (predictor) variables.

X = extracted_data.drop('target_5yrs', axis = 1)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about splitting your data into X and y](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/VxbUT/construct-a-naive-bayes-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# In `pandas`, subset your DataFrame by using square brackets `[]` to specify which column(s) to select.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Quickly subset a DataFrame to exclude a particular column by using the `drop()` function and specifying the column to drop.
# </details>

# ### Display the first 10 rows of your target data
# 
# Display the first 10 rows of your target and predictor variables. This will help you get a sense of how the data is structured.

# In[7]:


# Display the first 10 rows of your target data.

y.head(10)


# **Question:** What do you observe about the your target variable?
# 

# [Write your response here. Double-click (or enter) to edit.]

# In[8]:


# Display the first 10 rows of your predictor variables.

X.head(10)


# **Question:** What do you observe about the your predictor variables?

# [Write your response here. Double-click (or enter) to edit.]

# ### Perform a split operation on your data
# 
# Divide your data into a training set (75% of data) and test set (25% of data). This is an important step in the process, as it allows you to reserve a part of the data that the model has not observed. This tests how well the model generalizes—or performs—on new data.

# In[10]:


# Perform the split operation on your data.
# Assign the outputs as follows: X_train, X_test, y_train, y_test.

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about splitting your data between a training and test set](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/VxbUT/construct-a-naive-bayes-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the function in the `model_selection` module of `sklearn` on the features and target variable, in order to perform the splitting.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `model_selection.train_test_split()` function, passing in both `features` and `target`, while configuring the appropriate `test_size`.
# 
# Assign the output of this split as `X_train`, `X_test`, `y_train`, `y_test`.
# </details>

# ### Print the shape of each output 
# 
# Print the shape of each output from your train-test split. This will verify that the split operated as expected.

# In[13]:


# Print the shape (rows, columns) of the output from the train-test split.

# Print the shape of X_train.

print(X_train.shape)


# Print the shape of X_test.

print(X_test.shape)


# Print the shape of y_train.

print(y_train.shape)


# Print the shape of y_test.

print(y_test.shape)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Call the attribute that DataFrames in `pandas` have to get the number of rows and number of columns as a tuple.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `shape` attribute.
# </details>

# **Question:** How many rows are in each of the outputs?
# 

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What was the effect of the train-test split?
# 

# [Write your response here. Double-click (or enter) to edit.]

# ## Step 3: Model building

# **Question:** Which Naive Bayes algorithm should you use?

# [Write your response here. Double-click (or enter) to edit.]

# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about different implementations of the Naive Bayes](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/1zfDy/naive-bayes-classifiers) to determine which is appropriate in this situation.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Note that you are performing binary classification.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# You can identify the appropriate algorithm to use because you are performing a binary classification and assuming that the features of your model follow a normal distribution.
# </details>

# ### Fit your model to your training data and predict on your test data
# 
# By creating your model, you will be drawing on your feature engineering work by training the classifier on the `X_train` DataFrame. You will use this to predict `target_5yrs` from `y_train`.
# 
# Start by defining `nb` to be the relevant algorithm from `sklearn`.`naive_bayes`. Then fit your model to your training data. Use this fitted model to create predictions for your test data.

# In[14]:


# Assign `nb` to be the appropriate implementation of Naive Bayes.

### YOUR CODE HERE ###

nb = naive_bayes.GaussianNB()

# Fit the model on your training data.

### YOUR CODE HERE ###

nb.fit(X_train, y_train)

# Apply your model to predict on your test data. Call this "y_pred".

### YOUR CODE HERE ###

y_pred = nb.predict(X_test)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about constructing a Naive Bayes](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/VxbUT/construct-a-naive-bayes-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The appropriate implementation in this case is `naive_bayes`.`GaussianNB()`. Fit this model to your training data and predict on your test data.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `fit()`and pass your training feature set and target variable. Then call `predict()` on your test feature set.
# </details>

# ## Step 4: Results and evaluation
# 

# ### Leverage metrics to evaluate your model's performance
# 
# To evaluate the data yielded from your model, you can leverage a series of metrics and evaluation techniques from scikit-learn by examining the actual observed values in the test set relative to your model's prediction. Specifically, print the accuracy score, precision score, recall score, and f1 score associated with your test data and predicted values.

# In[15]:


# Print your accuracy score.

### YOUR CODE HERE ###

print('accuracy score:'), print(metrics.accuracy_score(y_test, y_pred))

# Print your precision score.

### YOUR CODE HERE ###

print('precision score:'), print(metrics.precision_score(y_test, y_pred))

# Print your recall score.

### YOUR CODE HERE ###

print('recall score:'), print(metrics.recall_score(y_test, y_pred))

# Print your f1 score.

### YOUR CODE HERE ###

print('f1 score:'), print(metrics.f1_score(y_test, y_pred))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about model evaluation](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/EITmV/key-evaluation-metrics-for-classification-models) for detail on these metrics.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `metrics` module in `sklearn` has a function for computing each of these metrics.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `accuracy_score()`, `precision_score()`, `recall_score()`, and `f1_score()`, passing `y_test`, and `y_pred` into each function.
# </details>

# **Question:** What is the accuracy score for your model, and what does this tell you about the success of the model's performance?
# 
# 

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** Can you evaluate the success of your model by using the accuracy score exclusively?
# 

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What are the precision and recall scores for your model, and what do they mean? Is one of these scores more accurate than the other?
# 

# [Write your response here. Double-click (or enter) to edit.]

# **Question:** What is the F1 score of your model, and what does this score mean?

# [Write your response here. Double-click (or enter) to edit.]

# ### Gain clarity with the confusion matrix
# 
# Recall that a confusion matrix is a graphic that shows your model's true and false positives and negatives. It helps to create a visual representation of the components feeding into the metrics.
# 
# Create a confusion matrix based on your predicted values for the test set.

# In[16]:


# Construct and display your confusion matrix.

# Construct the confusion matrix for your predicted and test values.

### YOUR CODE HERE ###

cm = metrics.confusion_matrix(y_test, y_pred)

# Create the display for your confusion matrix.

### YOUR CODE HERE ###

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)

# Plot the visual in-line.

### YOUR CODE HERE ###

disp.plot()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `metrics` module has functions to create a confusion matrix.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call `confusion_matrix`, passing in `y_test` and `y_pred`. Then, utilize `ConfusionMatrixDisplay()` to display your confusion matrix.
# </details>

# **Question:** What do you notice when observing your confusion matrix, and does this correlate to any of your other calculations?
# 

# [Write your response here. Double-click (or enter) to edit.]The top left to bottom right diagonal in the confusion matrix represents the correct predictions, and the ratio of these squares showcases the accuracy.
# 
# The concentration of true positives stands out relative to false positives. This ratio is why the precision score is so high (0.8406).
# 
# True negatives and false negatives are closer in number, which explains the worse recall score.

# ## Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# The evaluation of the model is important to inform if the model has delivered accurate predictions.
# Splitting the data was important for ensuring that there was new data for the model to test its predictive performance.
# Each metric provided an evaluation from a different standpoint, and accuracy alone was not a strong way to evaluate the model.
# Effective assessments balance the true/false positives versus true/false negatives through the confusion matrix and F1 score.
# 
# **How would you present your results to your team?**
# 
# [Write your response here. Double-click (or enter) to edit.]## Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# - The evaluation of the model is important to inform if the model has delivered accurate predictions.
# - Splitting the data was important for ensuring that there was new data for the model to test its predictive performance.
# - Each metric provided an evaluation from a different standpoint, and accuracy alone was not a strong way to evaluate the model. 
# - Effective assessments balance the true/false positives versus true/false negatives through the confusion matrix and F1 score.
# 
# **How would you present your results to your team?**
# - Showcase the data used to create the prediction and the performance of the model overall.
# - Review the sample output of the features and the confusion matrix to indicate the model's performance.
# - Highlight the metric values, emphasizing the F1 score.
# 
# **How would you summarize your findings to stakeholders?**
# - The model created provides some value in predicting an NBA player's chances of playing for five years or more.
# - Notably, the model performed better at predicting true positives than it did at predicting true negatives. In other words, it more accurately identified those players who will likely play for more than five years than it did those who likely will not.
# 
# 
# **How would you summarize your findings to stakeholders?**
# 
# [Write your response here. Double-click (or enter) to edit.]- The model created provides some value in predicting an NBA player's chances of playing for five years or more.
# - Notably, the model performed better at predicting true positives than it did at predicting true negatives. In other words, it more accurately identified those players who will likely play for more than five years than it did those who likely will not.
# 
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
