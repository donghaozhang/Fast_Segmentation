import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.digital_pathology_2018_loader import cellcancerLoader
from ptsemseg.loader.brats17_loader import Brats17Loader
from ptsemseg.loader.brats17_loader import Brats17Loader_guotai

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'camvid': camvidLoader,
        'ade20k': ADE20KLoader,
        'mit_sceneparsing_benchmark': MITSceneParsingBenchmarkLoader,
        'cellcancer': cellcancerLoader,
        'brats17_loader': Brats17Loader,
        'brats17_loader_guotai': Brats17Loader_guotai,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
