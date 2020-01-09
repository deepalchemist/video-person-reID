from .dataset import VideoDataset
from .mars import Mars
from .dukemtmcvidreid import DukeMTMCVidReID

__factory = {
    'mars': Mars,
    'duke': DukeMTMCVidReID,
}


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)
