import torch 
import ignite.distributed as idist
import importlib
    
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
    lr_scheduler = importlib.import_module("lr_scheduler." + config["lr_scheduler"]["name"])
    lr_scheduler = getattr(lr_scheduler, config["lr_scheduler"]["name"])
    lr_scheduler = lr_scheduler(optimizer, **config["lr_scheduler"]["args"])
    return lr_scheduler