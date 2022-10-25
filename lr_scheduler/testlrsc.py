from ignite.handlers import PiecewiseLinear
class testrlsc:
    def __init__(self,opt,config):
        self.opt = opt
        self.config = config
        parametername=config['lr_scheduler']['args']['parametername']
        milestones_values = eval(config['lr_scheduler']['args']['milestones_cmd'])
        self.lr_scheduler = PiecewiseLinear(self.opt, parametername, milestones_values)
        return self.lr_scheduler