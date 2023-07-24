from copy import deepcopy
import sys

sys.path.append('../')
from auton_survival import datasets
outcomes, features, id = datasets.load_support()

from auton_survival.preprocessing import Preprocessor

cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']

# Data should be processed in a fold-independent manner when performing cross-validation. 
# For simplicity in this demo, we process the dataset in a non-independent manner.
preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
x = preprocessor.fit_transform(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)



import numpy as np
times = np.quantile(outcomes.time[outcomes.event==1], [0.25, 0.5, 0.75]).tolist()

from auton_survival.experiments import SurvivalRegressionCV

param_grid = {'k' : [6],
              'distribution' : ['LogNormal'],
              'learning_rate' : [1e-2],
              'layers' : [[100]]}

experiment = SurvivalRegressionCV(model='dsm', num_folds=3, hyperparam_grid=param_grid, random_seed=2022)
# id = deepcopy(outcomes)
model = experiment.fit(x, outcomes,id, times, metric='brs')
print(experiment.folds)
# model
out_risk = model.predict_risk(x, times)
out_survival = model.predict_survival(x, times)
from auton_survival.metrics import survival_regression_metric

for fold in set(experiment.folds):
    print(survival_regression_metric('brs', outcomes[experiment.folds==fold], 
                                     out_survival[experiment.folds==fold], 
                                     times=times))


from auton_survival.metrics import survival_regression_metric

for fold in set(experiment.folds):
    print(survival_regression_metric('ctd', outcomes[experiment.folds==fold], 
                                     out_survival[experiment.folds==fold], 
                                     times=times))

for fold in set(experiment.folds):
    for time in times:
        print(time)
