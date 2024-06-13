"""
这个脚本的作用是定义两个函数，分别返回 网络模块 和 数据加载模块
"""
from typing import Dict

from modules.module.data_module import DataModule
from modules.module.net_module import NetModule

def fetch_data_module(dataset_name: str, dataset_config: Dict) -> DataModule:
    """
    :param dataset_name: 指定数据集的名称
    :param dataset_config: 数据加载模块的初始化 参数
    :return:
    """
    if dataset_name in {"gen1", "gen4", "DSEC", "NCaltech101"}:
        return DataModule(dataset_name=dataset_name, dataset_config=dataset_config)
    else:
        raise NotImplementedError

def fetch_net_module(network_config: Dict) -> NetModule:
    """
    :param network_config:
    :return:
    """
    return NetModule(network_config=network_config)
