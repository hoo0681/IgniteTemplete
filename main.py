import fire
import ignite.distributed as idist

from train import training
import yaml
def run(backend=None ,config_path='./config',**spawn_kwargs):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Parse config
    config["backend"] = backend
    
    with idist.Parallel(backend=config["backend"], **spawn_kwargs) as parallel:
        parallel.run(training, config)

if __name__ == "__main__":
    fire.Fire({"run": run})
