
import json

import numpy as np
import matplotlib.pyplot as plt

from skopt import BayesSearchCV

from skopt.space import Categorical, Integer
from skopt.plots import plot_evaluations, plot_objective, plot_convergence

from HyperparametersTuning.RNNHyperModel import RNNHyperModel

params = {
    'do_scale' : Categorical([True, False]),
    'type' : Categorical(['gru', 'lstm']),
    'hidden_size' : Integer(5, 250),
    'num_layers' : Integer(1, 20),
    'seq_len' : Integer(5, 100),
    'optimazer' : Categorical(['adam', 'rmsprop', 'nadam', 'adamw']),
    'use_batch_norm' : Categorical([True, False]),
    'dropout_rate' : Categorical([0, 0.1, 0.2, 0.3, 0.4, 0.5])
}

opt = BayesSearchCV(
    RNNHyperModel(),
    params,
    cv = 3,
    n_iter = 50,
    random_state = 1212
)

opt.fit(np.zeros(shape = 10), np.zeros(shape = 10))

fig1 = plt.figure()
plot_evaluations(opt)
plt.title('Parameter Evaluations')
plt.savefig('Output/parameter_evaluations.png')
plt.close(fig1)  

fig2 = plt.figure()
plot_objective(opt)
plt.title('Objective Function')
plt.savefig('Output/objective_function.png')
plt.close(fig2)

fig3 = plt.figure()
plot_convergence(opt)
plt.title('Convergence Plot')
plt.savefig('Output/convergence_plot.png')
plt.close(fig3)

with open('Output/best_params.json', 'w') as f:
    json.dump(opt.best_params_, f, indent = 6)
# end with

cv_results = {key: value.tolist() if isinstance(value, np.ndarray) else value 
              for key, value in opt.cv_results_.items()}

with open('Output/cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent = 6)
# end with
