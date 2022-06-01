import joblib
from DimensionReduction import *
from Decorator import time_indicator
from Config import dimension_reduction_methods
import os
import datetime


# TODO 修改为多线程并行
@time_indicator
def get_dim_data(dataset):

    if os.path.exists('dim_data/' + dataset):
        pass
    else:
        os.makedirs('dim_data/' + dataset)

    X = joblib.load('datasets/' + dataset + '.pkl')
    print('Data shape:', X.shape)

    # 三百六十度花式降维
    if '_tSNE_' in dimension_reduction_methods:
        tSNE_dim_data = t_SNE(X, dim=2, with_normalize=True)
        joblib.dump(tSNE_dim_data, 'dim_data/' + dataset + '/_tSNE_.pkl')

    if '_PCA_' in dimension_reduction_methods:
        PCA_dim_data, ratio, _ = get_pca(X, dim=20, with_normalize=True)
        joblib.dump(PCA_dim_data, 'dim_data/' + dataset + '/_PCA_.pkl')

    if '_FA_' in dimension_reduction_methods:
        FA_dim_data = feature_agglomeration(X, dim=20, with_normalize=True)
        joblib.dump(FA_dim_data, 'dim_data/' + dataset + '/_FA_.pkl')

    if '_UMAP_' in dimension_reduction_methods:
        UMAP_dim_data = get_umap(X, dim=20, with_normalize=True)
        joblib.dump(UMAP_dim_data, 'dim_data/' + dataset + '/_UMAP_.pkl')

    if '_LLE_' in dimension_reduction_methods:
        LLE_dim_data = get_lle(X, dim=20, with_normalize=True)
        joblib.dump(LLE_dim_data, 'dim_data/' + dataset + '/_LLE_.pkl')

    if '_MDS_' in dimension_reduction_methods:
        MDS_dim_data = get_mds(X, dim=20, with_normalize=True)
        joblib.dump(MDS_dim_data, 'dim_data/' + dataset + '/_MDS_.pkl')

    if '_Isomap_' in dimension_reduction_methods:
        Isomap_dim_data = get_Isomap(X, dim=20, with_normalize=True)
        joblib.dump(Isomap_dim_data, 'dim_data/' + dataset + '/_Isomap_.pkl')

    print('Dim Data Saved')


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
        get_dim_data(d, dimension_reduction_methods)

    end = datetime.datetime.now()
    print('总耗时', str(end - start))