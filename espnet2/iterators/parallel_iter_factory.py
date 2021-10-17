from espnet2.iterators.abs_iter_factory import AbsIterFactory

from typing import Iterable
def identity(x):
    return x


class ParallelIterFactory(AbsIterFactory):
    def __init__(self, *iter_factories, num_iters_per_epoch=None):
        self.iter_factories = iter_factories
        assert all(hasattr(item, 'num_iters_per_epoch') for item in self.iter_factories)
        assert all(hasattr(item, 'sampler') for item in self.iter_factories)
        if num_iters_per_epoch is None:
            num_iters_per_epoch = max(len(item.sampler) for item in self.iter_factories)
        for item in self.iter_factories:
            item.num_iters_per_epoch = num_iters_per_epoch
        self.collate_fn = self.iter_factories[0].collate_fn

        for item in self.iter_factories:
            item.collate_fn = identity


    def build_iter(self, epoch: int, shuffle: bool = None):
        iterators = [fact.build_iter(epoch=epoch, shuffle=shuffle) for fact in self.iter_factories]
        for item in zip(*iterators):
            rets = []; 
            for iitem in item: rets.extend(iitem)
            yield self.collate_fn(rets)


