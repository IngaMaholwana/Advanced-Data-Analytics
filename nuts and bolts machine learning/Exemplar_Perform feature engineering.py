#!/usr/bin/env python
# coding: utf-8

# # Exemplar: Perform feature engineering

# ## **Introduction**
# 
# 
# As you're learning, data professionals working on modeling projects use featuring engineering to help them determine which attributes in the data can best predict certain measures.
# 
# In this activity, you are working for a firm that provides insights to the National Basketball Association (NBA), a professional North American basketball league. You will help NBA managers and coaches identify which players are most likely to thrive in the high-pressure environment of professional basketball and help the team be successful over time.
# 
# To do this, you will analyze a subset of data that contains information about NBA players and their performance records. You will conduct feature engineering to determine which features will most effectively predict whether a player's NBA career will last at least five years. The insights gained then will be used in the next stage of the project: building the predictive model.
# 

# ## **Step 1: Imports** 

# Start by importing `pandas`.

# In[1]:


# Import pandas.

### YOUR CODE HERE ###

import pandas as pd


# The dataset is a .csv file named `nba-players.csv`. It consists of performance records for a subset of NBA players. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.

# Save in a variable named `data`.

### YOUR CODE HERE ###

data = pd.read_csv("nba-players.csv", index_col=0)


# <details><summary><h4><strong>Hint 1</strong></h4></summary>
# 
# The `read_csv()` function from `pandas` allows you to read in data from a csv file and load it into a DataFrame.
#     
# </details>

# <details><summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `read_csv()`, pass in the name of the csv file as a string, followed by `index_col=0` to use the first column from the csv as the index in the DataFrame.
#     
# </details>

# ## **Step 2: Data exploration** 

# Display the first 10 rows of the data to get a sense of what it entails.

# In[3]:


# Display first 10 rows of data.

### YOUR CODE HERE ###

data.head(10)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `pandas` library that can be called on a DataFrame to display the first n number of rows, where n is a number of your choice. 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call the `head()` function and pass in 10.
# </details>

# Display the number of rows and the number of columns to get a sense of how much data is available to you.

# In[4]:


# Display number of rows, number of columns.

### YOUR CODE HERE ###

data.shape


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# DataFrames in `pandas` have an attribute that can be called to get the number of rows and columns as a tuple.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# You can call the `shape` attribute.
# </details>

# **Question:** What do you observe about the number of rows and the number of columns in the data?
# - There are 1,340 rows and 21 columns in the data.

# Now, display all column names to get a sense of the kinds of metadata available about each player. Use the columns property in pandas.
# 

# In[5]:


# Display all column names.

### YOUR CODE HERE ###

data.columns


# The following table provides a description of the data in each column. This metadata comes from the data source, which is listed in the references section of this lab.
# 
# <center>
# 
# |Column Name|Column Description|
# |:---|:-------|
# |`name`|Name of NBA player|
# |`gp`|Number of games played|
# |`min`|Number of minutes played per game|
# |`pts`|Average number of points per game|
# |`fgm`|Average number of field goals made per game|
# |`fga`|Average number of field goal attempts per game|
# |`fg`|Average percent of field goals made per game|
# |`3p_made`|Average number of three-point field goals made per game|
# |`3pa`|Average number of three-point field goal attempts per game|
# |`3p`|Average percent of three-point field goals made per game|
# |`ftm`|Average number of free throws made per game|
# |`fta`|Average number of free throw attempts per game|
# |`ft`|Average percent of free throws made per game|
# |`oreb`|Average number of offensive rebounds per game|
# |`dreb`|Average number of defensive rebounds per game|
# |`reb`|Average number of rebounds per game|
# |`ast`|Average number of assists per game|
# |`stl`|Average number of steals per game|
# |`blk`|Average number of blocks per game|
# |`tov`|Average number of turnovers per game|
# |`target_5yrs`|1 if career duration >= 5 yrs, 0 otherwise|
# 
# </center>

# Next, display a summary of the data to get additional information about the DataFrame, including the types of data in the columns.

# In[6]:


# Use .info() to display a summary of the DataFrame.

### YOUR CODE HERE ###

data.info()


# **Question:** Based on the preceding tables, which columns are numerical and which columns are categorical?
# - Based on the preceding tables, the `name` column is categorical, and the rest of the columns are numerical.

# ### Check for missing values

# Now, review the data to determine whether it contains any missing values. Begin by displaying the number of missing values in each column. After that, use isna() to check whether each value in the data is missing. Finally, use sum() to aggregate the number of missing values per column.

# In[7]:


# Display the number of missing values in each column.
# Check whether each value is missing.
#Aggregate the number of missing values per column.

### YOUR CODE HERE ###

data.isna().sum()


# **Question:** What do you observe about the missing values in the columns? 
# All columns in this dataset have 0 missing values. 
# 

# **Question:** Why is it important to check for missing values?
# Checking for missing values is an important step in data exploration. Missing values are not particularly useful, so it's important to handle them by cleaning the data.

# ## **Step 3: Statistical tests** 

# Next, use a statistical technique to check the class balance in the data. To understand how balanced the dataset is in terms of class, display the percentage of values that belong to each class in the target column. In this context, class 1 indicates an NBA career duration of at least five years, while class 0 indicates an NBA career duration of less than five years.

# In[8]:


# Display percentage (%) of values for each class (1, 0) represented in the target column of this dataset.

### YOUR CODE HERE ###

data["target_5yrs"].value_counts(normalize=True)*100


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# In `pandas`, `value_counts(normalize=True)` can be used to calculate the frequency of each distinct value in a specific column of a DataFrame.  
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# After `value_counts(normalize=True)`, multipling by `100` converts the frequencies into percentages (%).
# </details>

# **Question:** What do you observe about the class balance in the target column?
# - About 62% of the values in the target columm belong to class 1, and about 38% of the values belong to class 0. In other words, about 62% of players represented by this data have an NBA career duration of at least five years, and about 38% do not. 
# - The dataset is not perfectly balanced, but an exact 50-50 split is a rare occurance in datasets, and a 62-38 split is not too imbalanced. However, if the majority class made up 90% or more of the dataset, then that would be of concern, and it would be prudent to address that issue through techniques like upsampling and downsampling. 

# **Question:** Why is it important to check class balance?
# 
# 
# If there is a lot more representation of one class than another, then the model may be biased toward the majority class. When this happens, the predictions may be inaccurate. 

# ## **Step 4: Results and evaluation** 
# 
# 
# Now, perform feature engineering, with the goal of identifying and creating features that will serve as useful predictors for the target variable, `target_5yrs`. 

# ### Feature Selection

# The following table contains descriptions of the data in each column:
# 
# <center>
# 
# |Column Name|Column Description|
# |:---|:-------|
# |`name`|Name of NBA player|
# |`gp`|Number of games played|
# |`min`|Number of minutes played|
# |`pts`|Average number of points per game|
# |`fgm`|Average number of field goals made per game|
# |`fga`|Average number of field goal attempts per game|
# |`fg`|Average percent of field goals made per game|
# |`3p_made`|Average number of three-point field goals made per game|
# |`3pa`|Average number of three-point field goal attempts per game|
# |`3p`|Average percent of three-point field goals made per game|
# |`ftm`|Average number of free throws made per game|
# |`fta`|Average number of free throw attempts per game|
# |`ft`|Average percent of free throws made per game|
# |`oreb`|Average number of offensive rebounds per game|
# |`dreb`|Average number of defensive rebounds per game|
# |`reb`|Average number of rebounds per game|
# |`ast`|Average number of assists per game|
# |`stl`|Average number of steals per game|
# |`blk`|Average number of blocks per game|
# |`tov`|Average number of turnovers per game|
# |`target_5yrs`|1 if career duration >= 5 yrs, 0 otherwise|
# 
# </center>

# **Question:** Which columns would you select and avoid selecting as features, and why? Keep in mind the goal is to identify features that will serve as useful predictors for the target variable, `target_5yrs`.  
# 
# - You should avoid selecting the `name` column as a feature. A player's name is not helpful in determining their career duration. Moreover, it may not be ethical or fair to predict a player's career duration based on a name.
# - The number of games a player has played in may not be as important in determining their career duration as the number of points they have earned. While you could say that someone who has played in more games may have more practice and experience, the points they earn during the games they played in would speak more to their performance as a player. This, in turn, would influence their career duration. So, the `gp` column on its own may not be a helpful feature. However, `gp` and `pts` could be combined to get the total number of points earned across the games played, and that result could be a helpful feature. That approach can be implemented later in the feature engineering process—in feature extraction. 
# - If the number of points earned across games will be extracted as a feature, then that could be combined with the number of minutes played across games (`min * gp`) to extract another feature. This could be a measure of players' efficiency and could help in predicting players' career duration. `min` on its own may not be useful as a feature for the same reason as `gp`.
# - There are three different columns that give information about field goals. The percent of field goals a player makes (`fg`) says more about their performance than the number of field goals they make (`fgm`) or the number of field goals they attempt (`fga`). The percent gives more context, as it takes into account both how many field goals a player successfully made and how many field goals they attempted in total. This allows for a more meaningful comparison between players. The same logic applies to the percent of three-point field goals made, as well as the percent of free throws made. 
# - There are columns for the number offensive rebounds (`oreb`), the number of defensive rebounds (`dreb`), and the number of rebounds overall (`reb`). Because the overall number of rebounds should already incorporate both offensive and defensive rebounds, it would make sense to use the overall as a feature. 
# - The number of assists (`ast`), steals (`stl`), blocks (`blk`), and turnovers (`tov`) also provide information about how well players are performing in games, and thus, could be helpful in predicting how long players last in the league. 
# 
# Therefore, at this stage of the feature engineering process, it would be most effective to select the following columns: 
# 
# `gp`, `min`, `pts`, `fg`, `3p`, `ft`, `reb`, `ast`, `stl`, `blk`, `tov`.

# Next, select the columns you want to proceed with. Make sure to include the target column, `target_5yrs`. Display the first few rows to confirm they are as expected.

# In[9]:


# Select the columns to proceed with and save the DataFrame in new variable `selected_data`.
# Include the target column, `target_5yrs`.

### YOUR CODE HERE ###

selected_data = data[["gp", "min", "pts", "fg", "3p", "ft", "reb", "ast", "stl", "blk", "tov", "target_5yrs"]]


# Display the first few rows.

### YOUR CODE HERE ###


selected_data.head()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the materials about feature selection and selecting a subset of a DataFrame.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use two pairs of square brackets, and place the names of the columns you want to select inside the innermost brackets. 
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# There is a function in `pandas` that can be used to display the first few rows of a DataFrame. Make sure to specify the column names with spelling that matches what's in the data. Use quotes to represent each column name as a string. 
# </details>

# ### Feature transformation

# An important aspect of feature transformation is feature encoding. If there are categorical columns that you would want to use as features, those columns should be transformed to be numerical. This technique is also known as feature encoding.

# **Question:** Why is feature transformation important to consider? Are there any transformations necessary for the features you want to use?
# - Many types of models are designed in a way that requires the data coming in to be numerical. So, transforming categorical features into numerical features is an important step. 
# - In this particular dataset, `name` is the only categorical column and the other columns are numerical (discussed in the exemplar response to Question 2). Given that `name` is not selected as a feature, all of the features that are selected at this point are already numerical and do not require transformation. 

# ### Feature extraction

# Display the first few rows containing containing descriptions of the data for reference. The table is as follows:
# 
# <center>
# 
# |Column Name|Column Description|
# |:---|:-------|
# |`name`|Name of NBA player|
# |`gp`|Number of games played|
# |`min`|Number of minutes played per game|
# |`pts`|Average number of points per game|
# |`fgm`|Average number of field goals made per game|
# |`fga`|Average number of field goal attempts per game|
# |`fg`|Average percent of field goals made per game|
# |`3p_made`|Average number of three-point field goals made per game|
# |`3pa`|Average number of three-point field goal attempts per game|
# |`3p`|Average percent of three-point field goals made per game|
# |`ftm`|Average number of free throws made per game|
# |`fta`|Average number of free throw attempts per game|
# |`ft`|Average percent of free throws made per game|
# |`oreb`|Average number of offensive rebounds per game|
# |`dreb`|Average number of defensive rebounds per game|
# |`reb`|Average number of rebounds per game|
# |`ast`|Average number of assists per game|
# |`stl`|Average number of steals per game|
# |`blk`|Average number of blocks per game|
# |`tov`|Average number of turnovers per game|
# |`target_5yrs`|1 if career duration >= 5 yrs, 0 otherwise|
# 
# </center>

# In[10]:


# Display the first few rows of `selected_data` for reference.

### YOUR CODE HERE ###

selected_data.head()


# **Question:** Which columns lend themselves to feature extraction?
# 
# - The `gp`, `pts`, `min` columns lend themselves to feature extraction.
#   - `gp` represents the total number of games a player has played in, and `pts` represents the average number of points the player has earned per game. It might be helpful to combine these columns to get the total number of points the player has earned across the games and use the result as a new feature, which could be added into a new column named `total_points`. The total points earned by a player can reflect their performance and shape their career longevity. 
#   - The `min` column represents the average number of minutes played per game. `total_points` could be combined with `min` and `gp` to extract a new feature: points earned per minute. This can be considered a measure of player efficiency, which could shape career duration. This feature can be added into a column named `efficiency`.

# Extract two features that you think would help predict `target_5yrs`. Then, create a new variable named 'extracted_data' that contains features from 'selected_data', as well as the features being extracted.

# In[11]:


# Extract two features that would help predict target_5yrs.
# Create a new variable named `extracted_data`.

### YOUR CODE HERE ### 

# Make a copy of `selected_data` 
extracted_data = selected_data.copy()

# Add a new column named `total_points`; 
# Calculate total points earned by multiplying the number of games played by the average number of points earned per game
extracted_data["total_points"] = extracted_data["gp"] * extracted_data["pts"]

# Add a new column named `efficiency`. Calculate efficiency by dividing the total points earned by the total number 
# of minutes played, which yields points per minute. (Note that `min` represents avg. minutes per game.)
extracted_data["efficiency"] = extracted_data["total_points"] / (extracted_data["min"] * extracted_data["gp"])

# Display the first few rows of `extracted_data` to confirm that the new columns were added.
extracted_data.head()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the materials about feature extraction.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function `copy()` to make a copy of a DataFrame. To access a specific column from a DataFrame, use a pair of square brackets and place the name of the column as a string inside the brackets.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use a pair of square brackets to create a new column in a DataFrame. The columns in DataFrames are series objects, which support elementwise operations such as multiplication and division. Be sure the column names referenced in your code match the spelling of what's in the DataFrame.
# </details>

# Now, to prepare for the Naive Bayes model that you will build in a later lab, clean the extracted data and ensure ensure it is concise. Naive Bayes involves an assumption that features are independent of each other given the class. In order to satisfy that criteria, if certain features are aggregated to yield new features, it may be necessary to remove those original features. Therefore, drop the columns that were used to extract new features.
# 
# **Note:** There are other types of models that do not involve independence assumptions, so this would not be required in those instances. In fact, keeping the original features may be beneficial.

# In[12]:


# Remove any columns from `extracted_data` that are no longer needed.

### YOUR CODE HERE ###

# Remove `gp`, `pts`, and `min` from `extracted_data`.
extracted_data = extracted_data.drop(columns=["gp", "pts", "min"])

# Display the first few rows of `extracted_data` to ensure that column drops took place.

### YOUR CODE HERE ###

extracted_data.head()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the materials about feature extraction.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There are functions in the `pandas` library that remove specific columns from a DataFrame and that display the first few rows of a DataFrame.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `drop()` function and pass in a list of the names of the columns you want to remove. By default, calling this function will result in a new DataFrame that reflects the changes you made. The original DataFrame is not automatically altered. You can reassign `extracted_data` to the result, in order to update it. 
# 
# Use the `head()` function to display the first few rows of a DataFrame.
# </details>

# Next, export the extracted data as a new .csv file. You will use this in a later lab. 

# In[13]:


# Export the extracted data.

### YOUR CODE HERE ###

extracted_data.to_csv("extracted_nba_players_data.csv", index=0)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `pandas` library that exports a DataFrame as a .csv file. 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `to_csv()` function to export the DataFrame as a .csv file. 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `to_csv()` function on `extracted_data`, and pass in the name that you want to give to the resulting .csv file. Specify the file name as a string and in the file name. Make sure to include `.csv` as the file extension. 
#     
#     Also, pass in the parameter `index` set to `0`, so that when the export occurs, the row indices from the DataFrame are not treated as an additional column in the resulting file. 
# </details>

# ## **Considerations**
# 
# 
# **What are some key takeaways that you learned during this lab?**
# - It is important to check for class balance in a dataset, particularly in the context of feature engineering and predictive modeling. If the target column in a dataset has more than 90% of its values belonging to one class, it is recommended to redistribute the data; otherwise, once a model is trained on the imbalanced data and predictions are made, the predictions may be biased. 
# - Feature selection involves choosing features that help predict the target variable and removing columns that may not be helpful for prediction. In this process, and throughout feature engineering, it is important to make ethical considerations.  
# - Feature transformation involves transforming features so that they are more usable for future modeling purposes, which includes encoding categorical features to turn them into numerical features. 
# - Feature extraction involves combining existing columns meaningfully to construct new features that would help improve prediction. 
# 
# **What summary would you provide to stakeholders? Consider key attributes to be shared from the data, as well as upcoming project plans.**
# - The following attributes about player performance could help predict their NBA career duration and should be included in a presentation to stakeholders: field goals, three-point field goals, free throws, rebounds, assists, steals, blocks, turnovers, total points, and efficiency as points per minute. 
# - It would be important to explain that these attributes, along with a relevant dataset, will be used in the next stage of the project. At that point, a model will be built to predict a player's career duration. Insights gained will be shared with stakeholders once the project is complete. Stakeholders would also appreciate being provided with a timeline and key deliverables that they can expect to receive.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
