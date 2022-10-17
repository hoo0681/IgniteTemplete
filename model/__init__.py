import importlib
import ignite.distributed as idist

def get_model(config):
    """
    Get model for training and validation.
    expected config:
    {
        ...
        "model": {
            "name": "<ModelName>",
            "args": <arg>, #model args {optional}
        }
        ...
    }
    """
    model = importlib.import_module("model." + config["model"]["name"])
    model = getattr(model, config["model"]["name"].lower())
    model = model(config['model']['args'])
    if idist.get_world_size() > 1:
        model = idist.auto_model(model)
    return model