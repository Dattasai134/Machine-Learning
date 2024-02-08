#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[13]:


df = pd.read_csv('canada_per_capita_income.csv')
df.head(6)


# In[17]:


plt.scatter(df['year'], df['per capita income (US$)'])
plt.xlabel('year')
plt.ylabel('rate')
plt.title('per capita income per year')
plt.show()


# In[19]:


X = df[['year']]
y = df['per capita income (US$)']


# In[20]:


# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)


# In[24]:


# Make predictions
y_pred = model.predict(X)
y_pred


# In[29]:


# Assuming model is the trained LinearRegression model and df is your DataFrame
# Extract the year 1970 and reshape it to a 2D array for prediction
prediction = [[1980]]

# Predict per capita income for the year 1970
predicted_income = model.predict(prediction)

# Display the predicted per capita income for 1970
print(f'Predicted per capita income: ${predicted_income[0]:,.2f}')


# In[ ]:




