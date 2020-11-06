#!/usr/bin/env python
# coding: utf-8

# # AUTHOR : Roshan Sibi
# **Data Science And Business Analytics Intern**
# 
# 
# **GRIP - The Sparks Foundation**

# Task 1 - Prediction using supervised machine learning.
# 
# I will predict the percentage of a student based on the no. of study hours.

# In[17]:


# Importing the libraries required

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the data from source

# In[18]:


# Reading data from remote link

url = "http://bit.ly/w-data"
dataset = pd.read_csv(url)
print("Dataset imported successfully")


# In[19]:


dataset.head(10)


# In[20]:


dataset.shape


# In[21]:


dataset.tail(5)


# data visualization

# In[22]:


# Plotting the data points on 2-D graph

dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# Preparing the data
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[23]:


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values  


# Splitting the data
# 
# The next step is to split the data into training and test sets. I'll do this by using Scikit-Learn's built-in train_test_split() method

# In[24]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0) 


# Training the Algorithm

# In[25]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# Plotting the regression line

# In[26]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()


# Making the predictions

# Now that the algorithm is trained, it is time to make some predictions on the score. The test data will be used

# In[27]:


print(X_test) 
y_pred = regressor.predict(X_test) 


# Comparing actual VS pedicted score

# In[28]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# Now let us answer the question in the task
# 
# **What will be the predicted score if astudent studies for 9.25 hours/day?**

# In[29]:


hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred))


# Evaluating the model

# In[30]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error= ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error=', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Conclusion
# 
# **I could successfully complete the predicyion using supervised ML task and could evaluate the model's performance on various parameters. THANK YOU!**

# In[ ]:




