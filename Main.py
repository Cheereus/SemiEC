from GetDimData import get_dim_data
from BaseClassify import base_classify
from IndicateVector import get_indicate_vector
from TrueRel import get_true_rel_mat
import os

DATA_PATH = 'datasets'
file_list = next(os.walk(DATA_PATH))[2]
datasets = [f.split('.')[0] for f in file_list if 'readme' not in f and 'labels' not in f and 'genes' not in f]

num_datasets = len(datasets)

# i = 0
# for d in datasets:
#     print('Dimension Reduction:', i, '/', num_datasets, d)
#     i +=1 
#     get_dim_data(d)

# i = 0
# for d in datasets:
#     print('Base Classify:', i, '/', num_datasets, d)
#     i +=1 
#     base_classify(d)

i = 0
for d in datasets:
    print('Indicate vector:', i, '/', num_datasets, d)
    i +=1 
    get_true_rel_mat(d)
    # get_indicate_vector(d)