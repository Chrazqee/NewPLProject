import os

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"


# config_path = "" -> 当前目录; config_name 指定解析哪个 yaml 文件
@hydra.main(config_path="", config_name="config", version_base="1.2")
def test_hydra_package(config: DictConfig):
    print("Type of config: ", type(config))  # Type of config:  <class 'omegaconf.dictconfig.DictConfig'>
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    loaded_yaml_path = "config.yaml"
    test_hydra_package()
