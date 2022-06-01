import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from Decorator import time_indicator
from Config import dimension_reduction_methods, cluster_methods


# @time_indicator
def k_means(X, k):
    k_m_model = KMeans(n_clusters=k, max_iter=300, n_init=40, init='k-means++')
    k_m_model.fit(X)
    return k_m_model.labels_.tolist()


@time_indicator
def knn(X, y, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X, y)
    return knn_model


# @time_indicator
def AGNES(X, k):
    labels = AgglomerativeClustering(n_clusters=k).fit(X).labels_
    return labels


# @time_indicator
def get_GMM(X, k):
    GMM = GaussianMixture(n_components=k, random_state=0).fit(X)
    labels = GMM.predict(X)
    return labels


def get_spectral(X, k):
    clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0).fit(X)
    return clustering.labels_

