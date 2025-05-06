import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("customer_data.csv")

# Preprocess
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply K-Means
kmeans = KMeans(n_clusters=4)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize
pca = PCA(n_components=2)
components = pca.fit_transform(scaled_data)
plt.scatter(components[:,0], components[:,1], c=df['Cluster'])
plt.title('Customer Segments')
plt.show()
