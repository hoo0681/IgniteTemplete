from pathlib import Path
from datetime import datetime
import ignite.distributed as idist
import ignite
def setup_logger(logger_config):
    """
    Setup logger for training and validation.
    expected config:
    {
        name: <logger_name>,
    }
    """
    #import logging
    #from logging import handlers
    #from ignite.utils import setup_logger
#
    #logger = logging.getLogger(name)
    #logger.setLevel(logging.INFO)
    #formatter = logging.Formatter(
    #    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #)
    #if distributed_rank <= 0:
    #    if filename is not None:
    #        handler = handlers.RotatingFileHandler(
    #            filename, maxBytes=10 * 1024 * 1024, backupCount=5
    #        )
    #    else:
    #        handler = logging.StreamHandler()
    #    handler.setFormatter(formatter)
    #    logger.addHandler(handler)
    logger=ignite.utils.setup_logger(logger_config['name'])
    return logger

def setup_rank_zero(logger, config):
    device = idist.device()

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = config["output_path"]
    folder_name = (
        f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    )
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    config["output_path"] = output_path.as_posix()
    logger.info(f"Output path: {config['output_path']}")

    if config["with_clearml"]:
        pass
        #from clearml import Task
        #task = Task.init("CIFAR10-Training", task_name=output_path.stem)
        #task.connect_configuration(config)
        ## Log hyper parameters
        #hyper_params = [
        #    "model",
        #    "batch_size",
        #    "momentum",
        #    "weight_decay",
        #    "num_epochs",
        #    "learning_rate",
        #    "num_warmup_epochs",
        #]
        #task.connect({k: v for k, v in config.items()})
