#!/usr/bin/env python
# coding: utf-8

# In[17]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\isrup\Downloads\Thesis2..csv")
# Read Data from csv file


# In[2]:


print(df.info()) 
# show the Information all of the column


# In[3]:


print(df)


# In[4]:


df.describe()


# In[16]:


csv_file_path = r"C:\Users\isrup\Downloads\Thesis2..csv"
print(numeric_columns)


# In[17]:


print(df.describe())


# In[18]:


df.isnull()
# Check if any data columns is null


# In[19]:


df.info()
# Check the data type 


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Importing PCA Library
scaler = StandardScaler()


# In[22]:


Res_data = pd.DataFrame(data=df['price'] , 
         columns= ['price'] + ['serve size'] + ['calories'] + ['Protein'] + ['Monthly Consumption'] + ['Revenue'])
Res_data['price'] = df['price']
Res_data['serve size'] = df['serve size']
Res_data['calories'] = df['calories']
Res_data['Protein'] = df['Protein']
Res_data['Monthly Consumption'] = df['Monthly Consumption']
Res_data['Revenue'] = df['Revenue']
# np.array

print(Res_data) # object Res_data
Res_data.info() # Checking to make sure Dtype are all Float


# In[23]:


Res_data.isnull() # make sure all the Data type is not null
print([Res_data])


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scaler to Fit into data :  vào dữ liệu
scaler.fit(Res_data)

# Performing Transform Scale for Res_data (5 columns) : 
# Thực hiện transform scale
cd = scaler.transform(Res_data)


# In[25]:


print(cd) 
# cd:  nd.array


# In[26]:


pca = PCA(n_components=2)  # Specify the desired number of components

# Impute missing values: Avoid array contains NaN values
# from which the PCA default cannot handle missing values encoded as NaN natively
# Mean Imputation scikit-learn library to impute missing values before scaling
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(cd)


# In[27]:


# Scale the imputed data
scaler = StandardScaler()
cd = scaler.fit_transform(data_imputed)
# succcesfully fit transform data cd


# In[28]:


# Apply PCA with n=2 variances from total 6 variances
pca = PCA(n_components=2) 

pca_result = pca.fit_transform(cd)


# In[29]:


print("Data Before Applying the PCA: ", cd.shape)
# Dữ liệu trc PCA: 


# In[30]:


print("Data After Applying the PCA:" , pca_result.shape)
# Dữ liệu sau PCA: 


# In[31]:


plt.figure(figsize = (12,12))
# First Comp:  Thành phần comp số 1
pca_1 = pca_result[:, 0]
# Second Comp: Thành phần comp số 2
pca_2 = pca_result[:, 1]


# Scatter Plot: Vẽ đồ thị
plt.scatter(x=pca_1, y = pca_2)
     


# In[32]:


Res_data['serve size']
# checking obj value


# In[33]:


plt.figure(figsize = (12,12))
# Comp 1
pca_1 = pca_result[:, 0]
# Comp 2
pca_2 = pca_result[:, 1]

# Draw a Scatter Plot
plt.scatter(x=pca_1, y = pca_2 ,c = Res_data['Monthly Consumption'])
# Highlighted plot show the Revenue data in specific Revenue,in which calculated by the formular : Price * Monthly Consumption
     


# In[34]:


pca.components_


# In[35]:


pca_comp = pd.DataFrame(data=pca.components_, columns= Res_data.columns)
# Comparison between comp 1 and 2 as well as how it correlated to the Res_data imported from df Originally
print(pca_comp) 


# In[ ]:





# In[ ]:





# In[ ]:




