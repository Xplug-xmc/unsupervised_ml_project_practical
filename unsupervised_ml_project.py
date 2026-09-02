#IMPORT THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#LOAD THE DATASET
df = pd.read_csv("Mall_Customers_Xplug.csv")

#INSPECT THE DATASET
print(f"\nFirst Five Rows:")
print(df.head())

print(f"\nDataset Information:")
print(df.info())

print(f"\nMissing Value:")
print(df.isnull().sum())

print(f"\n Dataset Summary:")
print(df.describe())


#SELECT THE FEATURES 
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


#VISUALIZE THE RAW DATASET
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Income vs Spending")
plt.show()


#FIND THE BEST NUMBER OF CLUSTERS USING THE ELBOW METHORD CODE
wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    wcss.append(model.inertia_)


#PLOT THE ELBOW
plt.plot(range(1, 11), wcss, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")

plt.show()

#BUILD THE FINAL K-MEAN MODEL
kmeans_model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

#TRAIN THE K-MEANS_MODEL
kmeans_model.fit(X)

#ASSIGN CUSTOMERS TO CLUSTERS
df["Cluster"] = kmeans_model.labels_

#DISPLAY THE CLUSTER ASSIGNMENTS
print(f"\nCustomer Cluster Assignments:")
print(df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())

#FIND THE CLUSTERED CENTER CENTERS
print(f"\n Cluster Centers:")
print(kmeans_model.cluster_centers_)

#PLOT THE FINAL CLUSTERS
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)


 #PLOT THE CENTROIDS
plt.scatter(
    kmeans_model.cluster_centers_[:, 0],
    kmeans_model.cluster_centers_[:, 1],
    color="black",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()