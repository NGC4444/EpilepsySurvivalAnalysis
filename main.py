
########################################

###    CV +  交叉验证  + 不要外部验证

#######################################

import sys
from auton_survival.models.dsm.load_nii import load_PET
sys.path.append('../')
from auton_survival import datasets

# img_id, images = load_PET()
import numpy as np
import torch
from auton_survival.preprocessing import Preprocessor
import random,os
import warnings
def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    

get_random_seed(2022)
warnings.filterwarnings('ignore')
outcomes, features, id = datasets.load_TLE200_new2('')
cat_feats = ['side', 'Sex', 'Freqmon', 'SE', 'SGS', 'early_brain_injury','familial_epilepsy','brain_hypoxia','Central_Nervous_System_Infections','traumatic_brain_injury','history_of_previous_surgery','MRI']
            # ['side', 'Sex', 'Freqmon', 'SE', 'SGS', 'early_brain_injury','familial_epilepsy','brain_hypoxia','Central_Nervous_System_Infections','traumatic_brain_injury','history_of_previous_surgery','MRI']
num_feats = ['Age', 'Onsetmon', 'Durmon']

# Data should be processed in a fold-independent manner when performing cross-validation. 
# For simplicity in this demo, we process the dataset in a non-independent manner.
preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
x = preprocessor.fit_transform(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)


import numpy as np
horizons = [12,24,36]
times = [12,24,36]
from auton_survival.experiments import SurvivalRegressionCV

param_grid = {'k' : [18],
              'distribution' : ['LogNormal', 'Weibull'],
              'learning_rate' : [1e-3],
              'layers' : [ [128]],
              "batch_size": [1]}
# param_grid = {'k' : [18],
#               'distribution' : ['Weibull'],
#               'learning_rate' : [1e-2],
#               'layers' : [ [128]],
#               "batch_size": [1]}
# param_grid = {'k' : [3],
#               'distribution' : ['Weibull'],
#               'learning_rate' : [1e-4, 1e-3,1e-2],
#               'layers' : [[100]]}

experiment = SurvivalRegressionCV(model='dsm', num_folds=5, hyperparam_grid=param_grid, random_seed=2022)
model = experiment.fit(x, outcomes, id, times,  metric='ctd')

# print(experiment.folds)
# print(model)



import torch
with torch.no_grad():

    out_risk = model.predict_risk(x, id, times, 'test')
    out_survival = model.predict_survival(x,id, times, 'test')
    np.save("result/pet+tab/mri_risk.npy",out_risk)
    np.save("result/pet+tab/mri_survival.npy",out_survival)

np.save("result/pet+tab/outcomes_mri.npy",outcomes)
np.save("result/pet+tab/folds.npy",experiment.folds)

from auton_survival.metrics import survival_regression_metric
print('========================')
print("      BScore            ")
print('========================')
bs = []
for fold in set(experiment.folds):
    
    bs.append(survival_regression_metric('brs', outcomes[experiment.folds==fold], 
                                     out_survival[experiment.folds==fold], 
                                     times=times))
print(bs)

from auton_survival.metrics import survival_regression_metric


print('========================')
print("      CIndex            ")
print('========================')
ci = []
for fold in set(experiment.folds):

    ci.append(survival_regression_metric('ctd', outcomes[experiment.folds==fold], 
                                     out_survival[experiment.folds==fold], 
                                     times=times))
print(ci)

from auton_survival.metrics import survival_regression_metric
print('========================')
print("         AUC            ")
print('========================')
ac = []
for fold in set(experiment.folds):
    try:
        ac.append(survival_regression_metric('auc', outcomes[experiment.folds==fold], 
                                        out_survival[experiment.folds==fold], 
                                        times=times))
    except:
        print("ValueError: censoring survival function is zero at one or more time points")
print(ac)

# for fold in set(experiment.folds):
#     for time in times:
#         print(time)