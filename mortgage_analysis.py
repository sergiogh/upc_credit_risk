import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# FreddieMac Family level loan dataset http://www.freddiemac.com/research/datasets/sf_loanlevel_dataset.page
df = pd.read_csv("sample_orig_2020.csv", sep="|", header=None, usecols=[0,5,7,8,9,10,12,16], names=['creditScore', 'insurancePercentage', 'occupancy', 'LTV', 'DebtToIncome', 'MortgateValue', 'interestRate', 'propertyState']) 

df=df.dropna()

#print(df.head())
print(df.describe())


#converting our projected array to pandas df

pca = df
pca.columns=['creditScore','DebtToIncome']
#build our algorithm with k=3, train it on pca and make predictions
kmeans = KMeans(n_clusters=3, random_state=0).fit(pca)
y_kmeans = kmeans.predict(pca)
#plotting the results 
plt.scatter(pca['First component'], pca['Second Component'], c=y_kmeans, s=50, alpha=0.5,cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50)