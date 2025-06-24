#!/usr/bin/env python
# coding: utf-8

# # Confidence intervals
# 
# Throughout the following exercises, you will learn to use Python to construct a confidence interval for a point estimate. Before starting on this programming exercise, we strongly recommend watching the video lecture and completing the IVQ for the associated topics.
# 
# 

# All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook. 

# As we move forward, you can find instructions on how to install required libraries as they arise in this notebook. Before we begin with the exercises and analyzing the data, we need to import all libraries and extensions required for this programming exercise. Throughout the course, we will be using numpy, pandas, and scipy stats for operations.

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats


# In[2]:


education_districtwise = pd.read_csv("education_districtwise.csv")
education_districtwise = education_districtwise.dropna()


# We’ll continue with our previous scenario, in which you’re a data professional working for the Department of Education of a large nation. Earlier, we imagined that the Department of Education asked you to collect the data on district literacy rates. You were only able to survey 50 randomly chosen districts, instead of all 634 districts included in your original dataset. You used Python to simulate taking a random sample of 50 districts, and make a point estimate of the population mean, or literacy rate for *all* districts. 

# Now imagine that the department asks you to construct a 95% confidence interval for your estimate of the mean district literacy rate. You can use Python to construct the confidence interval. 
# 

# You can also use the same sample data that you worked with earlier. Write the code to have Python simulate the same random sample of district literacy rate data. First, name your variable `sampled_data`. Then, enter the arguments of the `sample()` function. 
# 
# 
# *   `n`: Your sample size is `50`. 
# *   `replace`: Choose `True` because you are sampling with replacement.
# *   `random_state`: Choose the same random number to generate the same results–previously, you used `31,208`. 
# 
# 
# 
# 
# 
# 
#  
# 

# In[3]:


sampled_data = education_districtwise.sample(n=50, replace=True, random_state=31208)
sampled_data


# The output shows 50 districts selected randomly from your dataset. Each has a different literacy rate. 

# ## Construct a 95% confidence interval 
# 
# Now, construct a 95% confidence interval of the mean district literacy rate based on your sample data. Recall the four steps for constructing a confidence interval:
# 
# 1.   Identify a sample statistic
# 2.   Choose a confidence level
# 3.   Find the margin of error 
# 4.   Calculate the interval

# ### `scipy.stats.norm.interval()`
# 
# Earlier, you worked through these steps one by one to construct a confidence interval. With Python, you can construct a confidence interval with just a single line of code–and get your results faster! 
# 
# If you’re working with a large sample size, say larger than 30, you can construct a confidence interval for the mean using `scipy.stats.norm.interval()`. This function includes the following arguments: 
# 
# *   `alpha`: The confidence level
# *   `loc`: The sample mean
# *   `scale`: The sample standard error
# 
# **Reference**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html.
# 
# Let’s explore each argument in more detail. 

# 
# 
# #### `alpha`: The confidence level
# 
# The Department of Education requests a confidence level of 95%, which is the accepted standard for government funded research. 

# #### `loc`: The sample mean
# 
# This is the mean literacy rate of your sample of 50 districts. Name a new variable `sample_mean`. Then, compute the mean district literacy rate for your sample data. 

# In[4]:


sample_mean = sampled_data['OVERALL_LI'].mean()


# #### `scale`: The sample standard error
# 
# Recall that **standard error** measures the variability of your sample data. You may remember that the formula for the sample standard error is the sample standard deviation divided by the square root of the sample size.
# 
# **Note**: In practice, we typically don't know the true standard error, so we replace it with the estimated standard error.

# You can write code to express the formula and have Python do the calculation for you: 
# 
# 1. Name a new variable `estimated_standard_error`. 
# 2. Take the standard deviation of your sample data, and divide by the square root of your sample. 
# 3. In parentheses, write the name of your data frame followed by the shape function and zero in brackets. Recall that the shape function returns the number of rows and columns in a data frame. `shape[0]` returns only the number of rows, which is the same number as your sample size.  

# In[5]:


estimated_standard_error = sampled_data['OVERALL_LI'].std() / np.sqrt(sampled_data.shape[0])


# Now you’re ready to put all this together to construct your confidence interval for the mean using `scipy.stats.norm.interval()`. First, write out the function and set the arguments:
# 
# *   `alpha`: Enter `0.95` because you want to use a 95% confidence level
# *   `loc`: Enter the variable `sample_mean`
# *   `scale`: Enter the variable `estimated_standard_error`
# 

# In[6]:


stats.norm.interval(alpha=0.95, loc=sample_mean, scale=estimated_standard_error)


# You have a 95% confidence interval for the mean district literacy rate that stretches from about 71.4% to 77.0%. 
# 
# 95% CI: (71.42, 77.02)

# The Department of Education will use your estimate of the mean district literacy rate to help make decisions about distributing funds to different states.  

# ## Construct a 99% confidence interval 
# 
# Now imagine that a senior director in the department wants to be even *more* confident about your results. The director wants to make sure you have a reliable estimate, and suggests that you recalculate your interval with a 99% confidence level.
# 
# To compute a 99% confidence interval based on the same sample data, just change `alpha` to `0.99`. 
# 
# 

# In[7]:


stats.norm.interval(alpha=0.99, loc=sample_mean, scale=estimated_standard_error)


# You have a 99% confidence interval for the mean district literacy rate that stretches from about 70.5% to 77.9%.
# 
# 99% CI: (70.54, 77.90)

# ### Relationship between confidence level and confidence interval
# 
# You may notice that as the confidence *level* gets higher, the confidence *interval* gets wider. 
# 
# * With a confidence level of 95%, the interval covers 5.6 percentage points (71.4% - 77.0%)
# * With a confidence level of 99%, the interval covers 7.4 percentage points (70.5% - 77.9%)
# 
# This is because a wider confidence interval is more likely to include the actual population parameter.

# Your results will help the Department of Education decide how to distribute government resources to improve literacy. 

# If you have successfully completed the material above, congratulations! You now understand how to use Python to construct a confidence interval for a point estimate. Going forward, you can start using Python to construct confidence intervals for your own data.
