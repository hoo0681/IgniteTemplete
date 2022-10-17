import ignite.distributed as idist
from ignite.utils import manual_seed
from DAlogger import setup_logger,setup_rank_zero
from data import get_dataflow
from model import get_model
from optim import get_optimizer, get_lr_scheduler
from losses import get_criterion
from utils import log_basic_info, log_metrics, get_save_handler,create_trainer,create_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.contrib.engines import common


def training(local_rank, config):
    '''
    Training function used to define trainer and evaluator.
    expected config:
    {
        "seed": <seed>,
        "output_path": <output_path>,
        "num_iters_per_epoch": <num_iters_per_epoch>,
        "num_epochs": <num_epochs>,
        "validate_every": <validate_every>,
        "with_amp": <with_amp>,Bool
        "checkpoint_every": <checkpoint_every>,Int
        "resume_from": <resume_from>,String or None
        "log_every_iters": <log_every_iters>,Int
        "with_clearml": <with_clearml>,Bool

        "dataset": {
            "name": "<DatasetName>",
            "train_args": <train arg>, #train dataset args {optional}
            "val_args": <val arg>, #val dataset args {optional}
            "test_args": <test arg>, #test dataset args {optional}
            "batch_size": <batch_size>, #batch size 
            "num_workers": <num_workers>, #num workers 
        },
        "model": {
            "name": "<ModelName>",
            "args": <arg>, #model args {optional}
        },
        "optimizer": {
            "name": "<OptimizerName>",
            "args": <arg>, #optimizer args {optional}
        },
        "lr_scheduler": {
            "name": "<LRSchedulerName>",
            "args": <arg>, #lr_scheduler args {optional}
        },
        "loss": {
            "name": "<LossName>",
            "args": <arg>, #loss args {optional}
        },
        "logger": {
            "name": "<LoggerName>",
            "args": <arg>, #logger args {optional}
        },
    '''
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)

    logger = setup_logger(config['logger'])
    log_basic_info(logger, config)

    if rank == 0:
        setup_rank_zero(logger, config)

    train_loader, val_loader = get_dataflow(config['dataset'],test_only=False)
    model = get_model(config)
    optimizer = get_optimizer(config, model)
    criterion = get_criterion(config['loss'])
    config["num_iters_per_epoch"] = len(train_loader)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    trainer = create_trainer(
        model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger
    )

    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion),
    }

    train_evaluator = create_evaluator(model, metrics, config)
    val_evaluator = create_evaluator(model, metrics, config)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "train", state.metrics)
        state = val_evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "val", state.metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
        run_validation,
    )

    if rank == 0:
        evaluators = {"train": train_evaluator, "val": val_evaluator}
        tb_logger = common.setup_tb_logging(
            config["output_path"], trainer, optimizer, evaluators=evaluators
        )

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED,
        best_model_handler,
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        #wandb_logger.close()
        tb_logger.close()