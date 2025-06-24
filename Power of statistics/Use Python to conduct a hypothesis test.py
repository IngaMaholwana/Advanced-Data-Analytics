#!/usr/bin/env python
# coding: utf-8

# # Annotated follow-along guide: Use Python to conduct a hypothesis test
# 
# This notebook contains the code used in the following instructional video: [Use Python to conduct a hypothesis test](https://www.coursera.org/learn/the-power-of-statistics/lecture/FyATt/use-python-to-conduct-a-hypothesis-test).

# ## Introduction

# Throughout the following programming activity, you will learn to use Python to conduct a two-sample hypothesis test.  Before beginning the activity, watch the associated instructional video and complete the in-video question. All of the code you will be implementing and related instructions are contained in this notebook.

# ## Import packages and libraries
# 
# Before you begin the activity, import all the required libraries and extensions. Throughout the course, you will be using pandas and scipy stats for operations.

# In[1]:


import pandas as pd
from scipy import stats


# In[2]:


education_districtwise = pd.read_csv("education_districtwise.csv")
education_districtwise = education_districtwise.dropna()


# ## Activity overview

# This activity continues the scenario from an earlier part of the course, in which you are a data professional working for the Department of Education of a large nation. Recall that you are analyzing data on the literacy rate for each district.

# Now imagine that the Department of Education asks you to collect data on mean district literacy rates for two of the nation’s largest states: STATE21 and STATE28. STATE28 has almost 40 districts, and STATE21 has more than 70. Due to limited time and resources, you are only able to survey 20 randomly chosen districts in each state. The department asks you to determine if the difference between the two mean district literacy rates is statistically significant or due to chance. This will help the department decide how to distribute government funding to improve literacy. If there is a statistically-significant difference, the state with the lower literacy rate may receive more funding. 
# 
# In this activity, you will use Python to simulate taking a random sample of 20 districts in each state and conduct a two-sample t-test based on the sample data. 
# 

# ## Explore the data
# 
# To start, filter your dataframe for the district literacy rate data from the states STATE21 and STATE28. 
# 
# First, name a new variable: `state21`. Then, use the relational operator for equals (`==`) to get the relevant data from the `STATNAME` column. 

# In[4]:


state21 = education_districtwise[education_districtwise['STATNAME'] == "STATE21"]


# Next, name another variable: `state28`. Follow the same procedure to get the relevant data from the `STATNAME` column. 

# In[5]:


state28 = education_districtwise[education_districtwise['STATNAME'] == "STATE28"]


# ### Simulate random sampling
# 
# Now that you have organized your data, use the `sample()` function to take a random sample of 20 districts from each state. First, name a new variable: `sampled_state21`. Then, enter the arguments of the `sample()` function. 
# 
# *   `n`: Your sample size is `20`. 
# *   `replace`: Choose `True` because you are sampling with replacement.
# *   `random_state`: Choose an arbitrary number for the random seed – how about `13490`. 
# . 

# In[6]:


sampled_state21 = state21.sample(n=20, replace = True, random_state=13490)


# Now, name another variable: `sampled_state28`. Follow the same procedure, but this time choose a different number for the random seed; for example, 39,103. 

# In[7]:


sampled_state28 = state28.sample(n=20, replace = True, random_state=39103)


# ### Compute the sample means
# 
# You now have two random samples of 20 districts—one sample for each state. Next, use `mean()` to compute the mean district literacy rate for both STATE21 and STATE28.

# In[8]:


sampled_state21['OVERALL_LI'].mean()


# In[9]:


sampled_state28['OVERALL_LI'].mean()


# STATE21 has a mean district literacy rate of about 70.8%, while STATE28 has a mean district literacy rate of about 64.6%.
# 
# Based on your sample data, the observed difference between the mean district literacy rates of STATE21 and STATE28 is 6.2 percentage points (70.8% - 64.6%). 

# **Note**: At this point, you might be tempted to conclude that STATE21 has a higher overall literacy rate than STATE28. However, due to sampling variability, this observed difference might simply be due to chance, rather than an actual difference in the corresponding population means. A hypothesis test can help you determine whether or not your results are statistically significant. 

# ### Conduct a hypothesis test
# 
# Now that you’ve organized your data and simulated random sampling, you’re ready to conduct your hypothesis test. Recall that a two-sample t-test is the standard approach for comparing the means of two independent samples. To review, the steps for conducting a hypothesis test are:
# 
# 1.   State the null hypothesis and the alternative hypothesis.
# 2.   Choose a significance level.
# 3.   Find the p-value. 
# 4.   Reject or fail to reject the null hypothesis.

# #### Step 1: State the null hypothesis and the alternative hypothesis
# 
# The **null hypothesis** is a statement that is assumed to be true unless there is convincing evidence to the contrary. The **alternative hypothesis** is a statement that contradicts the null hypothesis and is accepted as true only if there is convincing evidence for it. 
# 
# In a two-sample t-test, the null hypothesis states that there is no difference between the means of your two groups. The alternative hypothesis states the contrary claim: there is a difference between the means of your two groups. 
# 
# We use $H_0$ to denote the null hypothesis and $H_A$ to denote the alternative hypothesis.
# 
# *   $H_0$: There is no difference in the mean district literacy rates between STATE21 and STATE28.
# *   $H_A$: There is a difference in the mean district literacy rates between STATE21 and STATE28.
# 
# 

# #### Step 2: Choose a significance level
# 
# The **significance level** is the threshold at which you will consider a result statistically significant. This is the probability of rejecting the null hypothesis when it is true. The Department of Education asks you to use their standard level of 5%, or 0.05.  

# #### Step 3: Find the p-value
# 
# **P-value** refers to the probability of observing results as or more extreme than those observed when the null hypothesis is true.
# 
# Based on your sample data, the difference between the mean district literacy rates of STATE21 and STATE28 is 6.2 percentage points. Your null hypothesis claims that this difference is due to chance. Your p-value is the probability of observing an absolute difference in sample means that is 6.2 or greater *if* the null hypothesis is true. If the probability of this outcome is very unlikely—in particular, if your p-value is *less than* your significance level of 5%— then you will reject the null hypothesis.

# #### `scipy.stats.ttest_ind()`
# 
# For a two-sample $t$-test, you can use `scipy.stats.ttest_ind()` to compute your p-value. This function includes the following arguments:
# 
# *   `a`: Observations from the first sample 
# *   `b`: Observations from the second sample
# *   `equal_var`: A boolean, or true/false statement, which indicates whether the population variance of the two samples is assumed to be equal. In our example, you don’t have access to data for the entire population, so you don’t want to assume anything about the variance. To avoid making a wrong assumption, set this argument to `False`. 
# 
# **Reference:** [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
# 

# Now write your code and enter the relevant arguments: 
# 
# *   `a`: Your first sample refers to the district literacy rate data for STATE21, which is stored in the `OVERALL_LI` column of your variable `sampled_ state21`.
# *   `b`: Your second sample refers to the district literacy rate data for STATE28, which is stored in the `OVERALL_LI` column of your variable `sampled_ state28`.
# *   `equal_var`: Set to `False` because you don’t want to assume that the two samples have the same variance.

# In[10]:


stats.ttest_ind(a=sampled_state21['OVERALL_LI'], b=sampled_state28['OVERALL_LI'], equal_var=False)


# Your p-value is about 0.0064, or 0.64%. 
# 
# This means there is only a 0.64% probability that the absolute difference between the two mean district literacy rates would be 6.2 percentage points or greater if the null hypothesis were true. In other words, it’s highly unlikely that the difference in the two means is due to chance.

# #### Step 4: Reject or fail to reject the null hypothesis
# 
# To draw a conclusion, compare your p-value with the significance level.
# 
# *   If the p-value is less than the significance level, you can conclude that there is a statistically significant difference in the mean district literacy rates between STATE21 and STATE28. In other words, you will reject the null hypothesis $H_0$.
# *   If the p-value is greater than the significance level, you can conclude that there is *not* a statistically significant difference in the mean district literacy rates between STATE21 and STATE28. In other words, you will fail to reject the null hypothesis $H_0$.
# 
# Your p-value of 0.0064, or 0.64%, is less than the significance level of 0.05, or 5%. Therefore, you will *reject* the null hypothesis and conclude that there is a statistically significant difference between the mean district literacy rates of the two states: STATE21 and STATE28. 

# Your analysis will help the Department of Education decide how to distribute government resources. Since there is a statistically significant difference in mean district literacy rates, the state with the lower literacy rate, STATE28, will likely receive more resources to improve literacy. 

# ## Conclusion
# 
# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
# 
# You now understand how to use Python to conduct a two-sample hypothesis test. Going forward, you can start using Python to conduct hypothesis tests on your own data.
