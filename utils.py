import torch 
import ignite.distributed as idist
import ignite
from pathlib import Path
import collections
from ignite.handlers import Checkpoint
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.engines import common

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def log_basic_info(logger, config):
    logger.info(f"Train on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in flatten(config).items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

def get_save_handler(config):
    """
    Setup model checkpointing
    expected config:
    {
        ...
        "output_path": <output_path>,
        "with_clearml": <with_clearml>,Bool
        ...
    }
    """
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return config["output_path"]
def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint

def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )

def create_trainer(
    model, optimizer, criterion, lr_scheduler, train_sampler, config, logger
):
    """
    expected config:
    {
        ...
        "with_amp": <with_amp>,Bool
        "checkpoint_every": <checkpoint_every>,Int
        "resume_from": <resume_from>,String or None
        "log_every_iters": <log_every_iters>,Int
        ...
    }
    내부적으로 위의 생성 단계 trainer는 다음과 같습니다.

    def train_step(engine, batch):

        x, y = batch[0], batch[1]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()

        with autocast(enabled=with_amp):
            y_pred = model(x)
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # If with_amp=False, this is equivalent to loss.backward()
        scaler.step(optimizer)  # If with_amp=False, this is equivalent to optimizer.step()
        scaler.update()  # If with_amp=False, this step does nothing

        return {"batch loss": loss.item()}

    trainer = Engine(train_step)

    """

    device = idist.device()
    amp_mode = None
    scaler = False
        
    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        device=device,
        non_blocking=True,
        output_transform=lambda x, y, y_pred, loss: {"batch loss": loss.item()},
        amp_mode="amp" if config["with_amp"] else None,
        scaler=config["with_amp"],
    )
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    if config["resume_from"] is not None:
        checkpoint = load_checkpoint(config["resume_from"])
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer
def create_evaluator(model, metrics, config):
    """
    expected config:
    {
        ...
        "with_amp": <with_amp>,Bool
        ...
    }
    """
    device = idist.device()
    amp_mode = "amp" if config["with_amp"] else None
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True, amp_mode=amp_mode
    )
    
    return evaluator