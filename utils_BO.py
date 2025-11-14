# import modules
import torch
torch.set_num_threads(1)
import numpy as np
import pandas as pd
import ConfigSpace as CS
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from yahpo_gym import BenchmarkSet
from math import sqrt


# numeric hp information
def extract_hp_info(space, cfg):
    '''
    Extract meta informations about numeric hyperparameter (only).
    Returns a dictionary with information about type, logscale, min & max of a hyperparameter.
    Only keeps hyperparameter that are in the initial cfg, to address conditionals search spaces.
    '''
    res = {}
    for name, hp in space.items():
        if name not in cfg.keys():
            continue
        if isinstance(hp, CS.hyperparameters.UniformFloatHyperparameter):
            res[name] = {
                'type': 'float',
                'log': hp.log,
                'min': hp.lower,
                'max': hp.upper,
            }
        elif isinstance(hp, CS.hyperparameters.UniformIntegerHyperparameter):
            res[name] = {
                'type': 'int',
                'log': hp.log,
                'min': hp.lower,
                'max': hp.upper,
            }
    return res


# Yahpo config --> tensor
def cfg_to_tensor(cfg: dict, hp_info: dict) -> torch.Tensor:
    '''
    Converts YAHPO configuration (only the numeric hyperparameters) into a (1, d) tensor.
    Requires meta information of the numeric hyperparamter.
    '''
    res = []
    for name, info in hp_info.items(): # only consider the numeric hyperparameters
        val = cfg.get(name)
        if val is None:
            raise ValueError(f'Missing value for hyperparameter "{name}" in cfg')
        if info.get('log', False):
            val = np.log(val)
        res.append(float(val))
    return torch.tensor([res], dtype=torch.double)


# tensor --> Yahpo config
def tensor_to_cfg(tensor: torch.Tensor, hp_info: dict) -> dict:
    '''
    Converts a (1, d) tensor into a YAHPO configuration (!numeric hp only!)
    Requires meta information of the numeric hyperparamter.
    '''
    x = tensor.squeeze().tolist()
    cfg = {}
    for i, (name, info) in enumerate(hp_info.items()):
        val = x[i]
        if info.get('log', False):
            val = np.exp(val)
        if info['type'] == 'int':
            val = int(round(val))
        cfg[name] = val
    return cfg


# train gp
def train_gp(x_train: torch.Tensor, y_train: torch.Tensor):
    '''
    Trains a gp using Botorch.
    Requires numeric part of the hyperparamter dictionary as Torch object.
    Requires accuracy vector as Torch object.
    Returns a trained model
    '''

    # initialize GP
    model = SingleTaskGP(x_train, y_train)

    # optimize mll
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


# optimize log EI & propose next candidate
def optimize_ei(model, y_train, hp_info, n_restarts=10, raw_samples=256):
    '''
    Optimizes the EI acquisition function and proposes next candidate to be evaluated.
    Requires a trained gp, a boundary torch object, numeric hyperparameter configurations.
    Returns next candidate configuration
    '''
    # incumbant performance
    best_f = y_train.max()

    # Acquisition Function
    ei = LogExpectedImprovement(model=model, best_f=best_f, maximize=True)

    # gather bounds out of hp_info
    lower_bo = []
    upper_bo = []

    for _, info in hp_info.items():
        lo = info['min']
        hi = info['max']

        # hp with log = T already live in log space -> log bounds as well
        if info.get('log', False):
            lo = np.log(lo)
            hi = np.log(hi)

        lower_bo.append(lo)
        upper_bo.append(hi)

    # concate to Torch object
    bounds = torch.tensor([lower_bo, upper_bo], dtype=torch.double)

    x_next, _ = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=n_restarts,
        raw_samples=raw_samples,
    )
    return x_next

# create valid candidate (fix conditional search space)
def sample_config(model, space):
    '''
    Samples a valid configuration from the YAHPO search space.
    Removes conditional hp by fixing them to a particular value.
    Fixes categorical hp to a specific value to simplify optimization.
    Returns a full valid configuration dictionary.
    Valid in this case means valid for this projects pipeline.
    '''
    cfg = space.sample_configuration().get_dictionary()
    
    # model specific adjustments needed
    if model == 'rbv2_xgboost':
        while (cfg.get('booster') != 'gbtree') or (cfg.get('num.impute.selected.cpo') != 'impute.median'): # fix conditional searchspace
            cfg = space.sample_configuration().get_dictionary()
    
    if model == 'rbv2_glmnet':
            while cfg.get('num.impute.selected.cpo') != 'impute.median': # fix imputation method
                cfg = space.sample_configuration().get_dictionary()

    return cfg


# run BO for a specific model, task & metric
def run_BO(model, metric, task_id, n):
    '''
    Runs BO on a YAHPO benchmark for a given model, task and metric.
    Does this n times independently.
    Returns a df containing the logged task, model, metric, iterations
    performance values, retrospective incumbent performance and all evaluated
    hp configurations.
    '''
    # YAHPO intialization
    bench = BenchmarkSet(model, task_id)
    space = bench.get_opt_space()

    # hp informations
    cfg = sample_config(model, space)
    hp_info = extract_hp_info(space, cfg)
    dim = len(hp_info)

    # config startingpoints
    n_s = 20

    ## BO Loop
    n_bo = round(20 + sqrt(40 * dim))
    res_df = pd.DataFrame()
    
    for j in range(n):
        
        # fresh empty hp df
        hp_df = pd.DataFrame()
        hp_start_df = pd.DataFrame()
        
        # fresh empty x and y tensors
        x_train = torch.empty((0, dim))
        y_train = torch.empty((0, 1))

        for _ in range(n_s):
            # random configuration
            cfg = sample_config(model, space)
            hp_start_df_row = pd.DataFrame([cfg])
            hp_start_df = pd.concat([hp_start_df, hp_start_df_row], ignore_index=True, axis = 0)

            # current configuration
            x = cfg_to_tensor(cfg, hp_info)
            x_train = torch.cat([x_train, x], dim=0)

            # accuracy
            obj = bench.objective_function(cfg)
            y_val = torch.tensor([[obj[0][metric]]], dtype=torch.double)

            # concat results
            y_train = torch.cat([y_train, y_val], dim=0)

        for _ in range(n_bo):
            # GP
            gp = train_gp(x_train, y_train)

            # next configurations
            x_next = optimize_ei(gp, y_train, hp_info)
            cfg_temp = tensor_to_cfg(x_next, hp_info)
            cfg_temp = {**cfg, **cfg_temp}
            cfg = cfg_temp

            # performance
            obj = bench.objective_function(cfg) # get accuracy of cfg
            y_val = torch.tensor([[obj[0][metric]]], dtype=torch.double)
            y_train = torch.cat([y_train, y_val], dim=0)

            # current configurations
            x = cfg_to_tensor(cfg, hp_info)
            x_train = torch.cat([x_train, x], dim=0)

            # log hp configurations
            hp_df_row = pd.DataFrame([cfg])
            hp_df = pd.concat([hp_df, hp_df_row], ignore_index=True, axis = 0)
        
        # result preparation
        iterations = [0]*n_s + list(range(1, n_bo+1))

        hp_df = pd.concat([hp_start_df, hp_df], ignore_index=True, axis = 0)
        
        res_df_temp = pd.concat([pd.DataFrame({
            'Model': model,
            'Inner_Iteration': iterations,
            'Metric': metric,
            'Performance': y_train.squeeze().tolist(),
            'Outer_Iteration': j+1
        }), hp_df], axis = 1)

        res_df_temp['Incumbant_Performance'] = res_df_temp['Performance'].cummax()
            
        res_df = pd.concat([res_df, res_df_temp], ignore_index=True, axis = 0)

    return res_df