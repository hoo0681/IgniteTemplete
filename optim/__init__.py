import torch 
import ignite.distributed as idist
import importlib

def get_optimizer(config,model):
    """
    Get optimizer for training.
    expected config:
    {
        ...
        "optimizer": {
            "name": "<OptimizerName>",
            "args": <arg>, #optimizer args {optional}
        }
        ...
    }
    """
    # search optimizer in folder first
    if hasattr(torch.optim, config["optimizer"]["name"]):
        optimizer = importlib.import_module("torch.optim." + config["optimizer"]["name"])
        print('using torch.optim')
    else:
        optimizer = importlib.import_module("optim." + config["optimizer"]["name"])
        optimizer = getattr(optimizer, config["optimizer"]["name"])
        print('using custom optim')
        
    optimizer = optimizer(model.parameters(), **config["optimizer"]["args"])
    optimizer = idist.auto_optim(optimizer)
    return optimizer

