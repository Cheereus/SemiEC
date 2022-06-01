from Config import dimension_reduction_methods, cluster_methods
import joblib
from Clustering import *
from Metrics import ARI, NMI, RelevanceMatrix
from Decorator import time_indicator
import os
import datetime

def base_classify(dataset):

    if os.path.exists('labels_pred/' + dataset):
        pass
    else:
        os.makedirs('labels_pred/' + dataset)

    if os.path.exists('rel_mat/' + dataset):
        pass
    else:
        os.makedirs('rel_mat/' + dataset)

    for dr in dimension_reduction_methods:
        for cl in cluster_methods:
            data = joblib.load('dim_data/' + dataset + '/' + dr + '.pkl')
            labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')
            labels_pred = None
            n_clusters = len(set(labels_true))

            if cl == 'kmeans':
                labels_pred = k_means(data, n_clusters)
            if cl == 'AGNES':
                labels_pred = AGNES(data, n_clusters)
            if cl == 'GMM':
                labels_pred = get_GMM(data, n_clusters)
            if cl == 'Spectral':
                labels_pred = get_spectral(data, n_clusters)

            # 计算并输出评价指标
            print('-------------')
            print(dataset, n_clusters, dr, cl)
            print('ARI:', ARI(labels_true, labels_pred))
            print('NMI:', NMI(labels_true, labels_pred))

            # 保存聚类结果，用于绘图和其他分析
            joblib.dump(labels_pred, 'labels_pred/' + dataset + '/' + dr + cl + '.pkl')

            # 生成相关矩阵并保存，用于后续处理
            rel_mat = RelevanceMatrix(labels_pred)
            joblib.dump(rel_mat, 'rel_mat/' + dataset + '/' + dr + cl + '.pkl')


if __name__ == '__main__':
    
    start = datetime.datetime.now()

    f_path = 'datasets'
    file_list = next(os.walk(f_path))[2]
    file_list = [f.split('.')[0] for f in file_list if 'readme' not in f and 'labels' not in f and 'genes' not in f]

    i = 1
    file_len = len(file_list)
    print(file_len)
    
    for d in file_list:
        print(i, '/', file_len, d)
        i +=1 
        base_classify(d, dimension_reduction_methods)

    end = datetime.datetime.now()
    print('总耗时', str(end - start))