#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
import pandas as pd
from sklearn.cluster import KMeans

# In[2]:
# Load your dataset into a DataFrame
data = pd.read_csv(r"C:\Users\isrup\Downloads\Thesis1.csv")
pd.set_option('display.max_columns', None)


# In[3]:
# Data Preprocessing
data['BigMac'] = pd.to_numeric(data['BigMac'])
data['BigMacCombo'] = pd.to_numeric(data['BigMacCombo'])
data['Filet-O-Fish'] = pd.to_numeric(data['Filet-O-Fish'])
data['Filet-O-Fish combo'] = pd.to_numeric(data['Filet-O-Fish combo'])
data['Small Fries'] = pd.to_numeric(data['Small Fries'])
data['Distance from root'] = pd.to_numeric(data['Distance from root'])


# In[4]:
# Calculate the correlation coefficient of the 4 items with the distance
correlation_bigmac = data['BigMac'].corr(data['Distance from root'])
correlation_bigmaccombo = data['BigMacCombo'].corr(data['Distance from root'])
correlation_filetofish = data['Filet-O-Fish'].corr(data['Distance from root'])
correlation_filetofish_combo = data['Filet-O-Fish combo'].corr(data['Distance from root'])


# In[5]:
# Print Correlation Coefficients
print(f"Correlation between BigMac and Distance from root: r = {correlation_bigmac:.4f}")
print(f"Correlation between BigMacCombo and Distance from root: r = {correlation_bigmaccombo:.4f}")
print(f"Correlation between Filet-O-Fish and Distance from root: r = {correlation_filetofish:.4f}")
print(f"Correlation between Filet-O-Fish combo and Distance from root: r = {correlation_filetofish_combo:.4f}")


# In[6]:
# Scatterplot for relationship of differnt item prices and the distance from the root store (Downtown Store)
fig = plt.figure(figsize=(10, 6))
plt.scatter(data['Distance from root'], data['BigMac'], label='BigMac', alpha=0.5)
plt.scatter(data['Distance from root'], data['BigMacCombo'], label='BigMacCombo', alpha=0.5)
plt.scatter(data['Distance from root'], data['Filet-O-Fish'], label='Filet-O-Fish', alpha=0.5)
plt.scatter(data['Distance from root'], data['Filet-O-Fish combo'], label='Filet-O-Fish combo', alpha=0.5)
plt.scatter(data['Distance from root'], data['Small Fries'], label='Small Fries', alpha=0.5)
plt.xlabel('Distance from Root')
plt.ylabel('Price')
plt.title('Price by Distance from Root for Different Items')
plt.legend()
plt.show()


# In[7]:
# Extract the relevant columns for clustering
food_items = ['BigMac', 'BigMacCombo', 'Filet-O-Fish', 'Filet-O-Fish combo', 'Small Fries']

# Set up the grid of subplots
fig, axes = plt.subplots(1, len(food_items), figsize=(18, 6))
fig.suptitle('K-means Clustering of Food Item Prices', fontsize=16)

# Perform k-means clustering for each food item and create subplots
for idx, item in enumerate(food_items):
    # Extract the data for the current food item
    food_data = data[['Distance from root', item]]
    
    # Perform k-means clustering
    k = 3  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(food_data)
    
    # Add the cluster labels to the DataFrame
    food_data['Cluster'] = clusters
    
    # Create scatter plot in the corresponding subplot
    sns.scatterplot(data=food_data, x='Distance from root', y=item, hue='Cluster', palette='Set1', ax=axes[idx])
    axes[idx].set_xlabel('Distance from Root')
    axes[idx].set_ylabel(item + ' Price')
    axes[idx].set_title(f'Clustering: {item}')
    axes[idx].legend(title='Cluster')

plt.tight_layout()
plt.show()







# In[ ]:





# In[ ]:




