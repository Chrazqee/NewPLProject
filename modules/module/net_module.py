"""
此 dataset_config 的作用是构建一个 Module 类，该类继承 pl.LightningModule
该类需要重载 父类 中的一些方法
- def setup(self, stage: Optional[str] = None): ...
- def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT: ...
- def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]: ...
- def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]: ...
- def on_train_epoch_end(self) -> None: ...
- def on_validation_epoch_end(self) -> None: ...
- def on_test_epoch_end(self) -> None: ...
- def configure_optimizers(self) -> Any: ...
"""
from typing import Any, Optional, Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class NetModule(pl.LightningModule):
    def __init__(self, network_config: Dict):
        super().__init__()
        pass

    def setup(self, stage: str) -> None:
        """
        一般用于定义各种参数
        :param stage:
        :return:
        """
        if stage == "fit":
            pass

        elif stage == "validation":
            pass

        elif stage == "test":
            pass

        else:
            raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:  # STEP_OUTPUT = Union[Tensor, Dict[str, Any]]
        """
        定义前向过程
        :param batch:
        :param batch_idx:
        :return: 返回的东西会传给 callback; loss 也在这个位置计算
        """
        data = batch[0]  # (batch_size, 2, 3, 224, 224)  输入数据，用于网络训练
        target = batch[1]  # [{"bbox": , "label": }, {"bbox": , "label": }, ...]  groundtruth, 用于计算 loss
        extra_for_vis = batch[2]  # 用于可视化的额外信息

        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:

        pass

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:

        pass

    def configure_optimizers(self) -> Any:
        pass
