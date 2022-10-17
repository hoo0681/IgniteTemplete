import torch.nn as nn
import ignite.distributed as idist
import importlib

def get_criterion(loss_config):
    """
    Get criterion for training and validation.
    expected loss_config:
    {
        "name": "<LossName>",
        "args": <arg>, #loss args {optional}
    }
    """
    criterion = importlib.import_module("losses." + loss_config["name"])
    criterion = getattr(criterion, loss_config["name"].lower())
    criterion = criterion(loss_config['args']).to(idist.device())
    #if idist.get_world_size() > 1:
    #    criterion = idist.auto_model(criterion)
    return criterion