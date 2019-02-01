import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse

from data_loader import DataLoader
from primitive_helpers import bike_human_distance, bike_human_size, bike_human_position, bike_human_nums

from snorkel.learning import GenerativeModel
from metal.label_model import LabelModel
from metal.tuners.hyperband_tuner import HyperbandTuner


loader = DataLoader('/data/')

def has_bike(object_names):
    if ('cycle' in object_names) or ('bike' in object_names) or ('bicycle' in object_names):
        return 1
    else:
        return 0

def has_human(object_names):
    if (('person' in object_names) or ('woman' in object_names) or ('man' in object_names)) \
        and (('bicycle' in object_names) or 'bicycles' in object_names):
        return 1
    else:
        return 0

def LF_position(bike_human_position):
    return bike_human_position

def LF_distance(has_human, has_bike, bike_human_distance):
    if has_human >= 1:
        if has_bike >= 1: 
            if bike_human_distance <= 20:
                return 1
            else:
                return 0
    else:
        return -1
    
def LF_size(has_human, has_bike, bike_human_size):
    if has_human >= 1:
        if has_bike >= 1: 
            if bike_human_size <= 50:
                return -1
            else:
                return 0
    else:
        return -1
      
def LF_number(has_human, has_bike, bike_human_num):
    if has_human >= 1:
        if has_bike >= 1: 
            if bike_human_num >= 2:  # more human than bikes
                return 1
            if bike_human_num >= 1:
                return 0
            if bike_human_num >= 0:
                return 1
    else:
        return -1


def get_L_matrix(num, object_names, object_x, object_y, object_width, object_height, ground,
                 print_stats_table=False):
    m = 6  # number of primitives
    primitive_mtx = np.zeros((num,m))
    for i in range(num):
        primitive_mtx[i,0] = has_human(object_names[i])
        primitive_mtx[i,1] = has_bike(object_names[i])
        primitive_mtx[i,2] = bike_human_distance(object_names[i], 
                                                 object_x[i], 
                                                 object_y[i])

        area = np.multiply(object_height[i], object_width[i])
        primitive_mtx[i,3] = bike_human_size(object_names[i], area)
        primitive_mtx[i,4] = bike_human_position(object_names[i], 
                                                 object_x[i], 
                                                 object_y[i])
        primitive_mtx[i,5] = bike_human_nums(object_names[i])
    p_keys = {
        'has_human': primitive_mtx[:,0],
        'has_bike': primitive_mtx[:, 1],
        'bike_human_distance': primitive_mtx[:, 2],
        'bike_human_size': primitive_mtx[:, 3],
        'bike_human_position': primitive_mtx[:, 4],
        'bike_human_num': primitive_mtx[:, 5]
    }
    L_fns = [LF_position,LF_distance,LF_size,LF_number]
    L = np.zeros((len(L_fns),num)).astype(int)
    for i in range(num):
        L[0,i] = L_fns[0](p_keys['bike_human_position'][i])
        L[1,i] = L_fns[1](p_keys['has_human'][i], p_keys['has_bike'][i], p_keys['bike_human_distance'][i])
        L[2,i] = L_fns[2](p_keys['has_human'][i], p_keys['has_bike'][i], p_keys['bike_human_size'][i])
        L[3,i] = L_fns[3](p_keys['has_human'][i], p_keys['has_bike'][i], p_keys['bike_human_num'][i])

    if print_stats_table:
        stats_table = np.zeros((len(L),2))
        total = float(num)
        for i in range(len(L)):
            # coverage: (num labeled) / (total)
            stats_table[i,0] = np.sum(L[i,:] != 0)/ total
            
            # accuracy: (num correct assigned labels) / (total assigned labels)
            stats_table[i,1] = np.sum(L[i,:] == ground)/float(np.sum(L[i,:] != 0))

        stats_table = pd.DataFrame(stats_table, index = [lf.__name__ for lf in L_fns], columns = ["Coverage", "Accuracy"])
        print (stats_table)
    L_sparse = sparse.csr_matrix(L.T)
    return L.T, L_sparse


L_train, L_train_sparse = get_L_matrix(
    loader.train_num, loader.train_object_names, loader.train_object_x, loader.train_object_y,
    loader.train_object_width, loader.train_object_height, loader.train_ground, print_stats_table=True)

L_val, L_val_sparse = get_L_matrix(
    loader.val_num, loader.val_object_names, loader.val_object_x, loader.val_object_y,
    loader.val_object_width, loader.val_object_height, loader.val_ground, print_stats_table=True)

metrics = ["accuracy", "precision", "recall", "f1"]

# ####### Majority Vote ########
# mv_labels = np.sign(np.sum(L.T,1))
# print ('Coverage of Majority Vote on Train Set: ', np.sum(np.sign(np.sum(np.abs(L.T),1)) != 0)/float(loader.train_num))
# print ('Accuracy of Majority Vote on Train Set: ', np.sum(mv_labels == loader.train_ground)/float(loader.train_num))


########################
####### Snorkel ########
########################
print ('\n\n\n####### Running Snorkel Generative Model ########')
gen_model = GenerativeModel()
gen_model.train(L_train, epochs=100, decay=0.95, step_size= 0.01/ L_train.shape[0], reg_param=1e-6)
print(gen_model.score(L_train_sparse, loader.train_ground))

######################
####### METAL ########
######################

# remap labels so that they are in {1, 2}
def remap_labels(data):
    transformed_data = np.zeros(data.shape, dtype=np.int)
    transformed_data[data == -1] = 1
    transformed_data[data == 1] = 2
    return transformed_data

train_ground = remap_labels(loader.train_ground)
val_ground = remap_labels(loader.val_ground)
L_train_sparse = sparse.csc_matrix((remap_labels(L_train_sparse.data), L_train_sparse.indices, L_train_sparse.indptr)).T
L_val_sparse = sparse.csc_matrix((remap_labels(L_val_sparse.data), L_val_sparse.indices, L_val_sparse.indptr)).T


print ('\n\n####### Running METAL Label Model ########')
label_model = LabelModel()
label_model.train_model(L_train_sparse, n_epochs=200, print_every=50, seed=123, verbose=False)
train_marginals = label_model.predict_proba(L_train_sparse)
label_model.score((L_train_sparse, train_ground), metric=metrics)


####### METAL with Exact Class Balance ########
print ('\n\n####### Running METAL Label Model with exact class balance ########')
train_class_balance = np.array([
    np.sum(train_ground == 1) / loader.train_num,
    np.sum(train_ground == 2) / loader.train_num
    ])
val_class_balance = np.array([
    np.sum(val_ground == 1) / loader.val_num,
    np.sum(val_ground == 2) / loader.val_num
    ])
print ('Train set class balance:', train_class_balance)
print ('Val set class balance:', val_class_balance)
label_model2 = LabelModel(seed=123)
label_model2.train_model(L_train_sparse, class_balance=train_class_balance, n_epochs=500, verbose=False)
train_marginals = label_model2.predict_proba(L_train_sparse)
label_model2.score((L_train_sparse, train_ground), metric=metrics)


####### METAL with Class Balance ########
print ('\n\n####### Running METAL Label Model with estimated class balance ########')
class_balance = np.array([0.9, 0.1])
print ('Class balance:', class_balance)
label_model3 = LabelModel(seed=123)
label_model3.train_model(L_train_sparse, class_balance=class_balance, n_epochs=500, verbose=False)
train_marginals = label_model3.predict_proba(L_train_sparse)
label_model3.score((L_train_sparse, train_ground), metric=metrics)
# train_marginals = np.array([t[0] for t in train_marginals]) # the probabilities that the label is 1.



####### Hyperband Tuning #######
print ('\n\n####### Tuning METAL Label Model ########')
class_balance=[.99, .01]
print ('Class balance:', class_balance)
search_space = {
    'seed' : [123],
    'n_epochs': {'range': [10,100]},
    'lr': {'range': [1e-5, 5e-2]},
    'momentum': {'range': [1e-5, 1]},
    'print_every': 5,
    'prec_init': {'range': [1e-5, 1]},
    'l2': {'range': [0, 1]}
}
hb_tuner = HyperbandTuner(LabelModel, hyperband_epochs_budget=10000, seed=123, validation_metric="precision")
train_args = [L_train_sparse, None, [], class_balance]  # L_train, Y_dev, deps, class_balance
best_hb_model = hb_tuner.search(search_space, (L_val_sparse, val_ground), train_args=train_args, verbose=False, seed=123)

####### Use Tuned Parameters #######
print ('\n\n####### Running Tuned METAL Label Model ########')
best_hb_model.train_model(L_train_sparse, class_balance=class_balance, verbose=False)
train_marginals = best_hb_model.predict_proba(-L_train_sparse)
best_hb_model.score((L_train_sparse, train_ground), metric=metrics)
# train_marginals = np.array([t[0] for t in train_marginals]) # the probabilities that the label is 1.
