from importlib import import_module
import torch
def get_dataflow(dataset_config,test_only=False):
    """
    Get dataflow for training and validation.
    expected dataset_config:
    {
        "name": "<DatasetName>",
        "train_args": <train arg>, #train dataset args {optional}
        "val_args": <val arg>, #val dataset args {optional}
        "test_args": <test arg>, #test dataset args {optional}
        "batch_size": <batch_size>, #batch size 
        "num_workers": <num_workers>, #num workers 
    }
    """
    if ~test_only:            
        train_dataset = import_module("data." + dataset_config["name"])
        train_dataset = getattr(train_dataset, dataset_config["name"].lower())
        train_dataset = train_dataset(dataset_config['train_args'], train=True)

        val_dataset = import_module("data." + dataset_config["name"])
        val_dataset = getattr(val_dataset, dataset_config["name"].lower())
        val_dataset = val_dataset(dataset_config['val_args'], train=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataset_config["batch_size"],
            shuffle=True,
            num_workers=dataset_config["num_workers"],
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=dataset_config["batch_size"],
            shuffle=False,
            num_workers=dataset_config["num_workers"],
            pin_memory=True,
        )
        return train_loader, val_loader
    else:
        test_dataset = import_module("data." + dataset_config["name"])
        test_dataset = getattr(test_dataset, dataset_config["name"].lower())
        test_dataset = test_dataset(dataset_config['test_args'], train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=dataset_config["batch_size"],
            shuffle=False,
            num_workers=dataset_config["num_workers"],
            pin_memory=True,
        )
        return test_loader
