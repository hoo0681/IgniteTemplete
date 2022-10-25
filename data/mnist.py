from torchvision import datasets, models
from torchvision.transforms import (
    Compose,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

class mnist(datasets.MNIST):
    def __init__(self,**kwargs):
        root=kwargs['root']
        download=kwargs['download']
        train=kwargs['train']
        super().__init__(self,root, train, download)
        self.transform = Compose([
                ToTensor() 
            ])

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = self.transform(img)
        return img, target