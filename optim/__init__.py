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
    optimizer = importlib.import_module("torch.optim." + config["optimizer"]["name"])
    optimizer = getattr(optimizer, config["optimizer"]["name"])
    optimizer = optimizer(model.parameters(), **config["optimizer"]["args"])
    optimizer = idist.auto_optim(optimizer)
    return optimizer
def get_lr_scheduler(config, optimizer):
    """
    Get lr scheduler for training.
    expected config:
    {
        ...
        "lr_scheduler": {
            "name": "<LRSchedulerName>",
            "args": <arg>, #lr_scheduler args {optional}
        }
        ...
    }
    """
    lr_scheduler = importlib.import_module("torch.optim.lr_scheduler." + config["lr_scheduler"]["name"])
    lr_scheduler = getattr(lr_scheduler, config["lr_scheduler"]["name"])
    lr_scheduler = lr_scheduler(optimizer, **config["lr_scheduler"]["args"])
    return lr_scheduler