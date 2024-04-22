"""
Saves or loads model dictionary as well as parameters.
"""

import os
import uuid
import pickle
import torch

def _state_dict_filename(model_uuid_str: str) -> str:
    return model_uuid_str+".pt"

def _params_filename(model_uuid_str: str) -> str:
    return model_uuid_str+".pk"

def save_model(state_dict, directory,  env, *vs) -> str:
    """
    Saves model state dict and user-defined parameters from env (e.g. local()) to files.
    Model state dict is saved to {UUID}.pt file
    User defined parameters in *vs can be saved to {UUID}.pk file
    """
    assert os.path.isdir(directory), f"{directory} is not a directory"
    model_uuid_str = str(uuid.uuid4())
    torch.save(state_dict, os.path.join(directory, _state_dict_filename(model_uuid_str)))

    params_dict = dict([(x, env[x]) for v in vs for x in env if v is env[x]])
    with open( os.path.join(directory, _params_filename(model_uuid_str)), 'wb') as f:
        pickle.dump(params_dict, f)
    return model_uuid_str

def load_model(directory, model_uuid_str, env):
    """
    Reads model state_dict from file {UUID}.pt file.
    Loads user-defined variables in {UUID}.pk to environment (e.g., local())
    """
    assert os.path.isdir(directory), f"{directory} is not a directory"
    full_model_path = os.path.join(directory, _state_dict_filename(model_uuid_str))
    full_param_path = os.path.join(directory, _params_filename(model_uuid_str))
    assert os.path.isfile(full_model_path), f"model file {full_model_path} does not exist."
    assert os.path.isfile(full_param_path), f"param file {full_param_path} does not exist."
    state_dict = torch.load(full_model_path)
    with open(full_param_path, 'rb') as f:
        d = pickle.load(f)
        env.update(d)
    return state_dict