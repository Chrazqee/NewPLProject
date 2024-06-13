import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from modules.module_fetch import fetch_data_module, fetch_net_module


# config_path: 指定 yaml 文件在哪个目录下
@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    # 将 config 打印出来看看
    print(OmegaConf.to_yaml(config))

    # 定义训练策略
    distributed_backend = config.training.hardware.dist_backend
    gpus = config.training.hardware.gpus
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    dataset_config = config.dataset_config
    assert dataset_config is not None
    dataset_name = config.dataset_name
    assert dataset_name is not None
    data_module = fetch_data_module(dataset_name=dataset_name,
                                    dataset_config=dataset_config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    # 🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗 `ckpt_path` 🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗
    ckpt_path = None
    SAVE_DIR = "./TB_RECODE"
    MODEL_NAME = "SNN"
    VERSION = "V1"
    tb_logger = TensorBoardLogger(save_dir=SAVE_DIR,
                                  name=MODEL_NAME,
                                  version=VERSION,  # TODO: 如果 ckpt_path 是空，到下一个版本，如果不为空，说明在断点续训
                                  log_graph=False)
    tb_logger.log_hyperparams(config)  # 记录超参数

    # ---------------------
    # Model
    # ---------------------
    network_config = config.network_config
    assert network_config is not None
    net_module = fetch_net_module(network_config=network_config)

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(
            logging_interval="step"  # 'epoch' or 'step'
        ))

    # ---------------------
    # Training
    # ---------------------
    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch  # 1
    assert val_check_interval is None or check_val_every_n_epoch is None  # 其中之一需要是 None
    # 指定各种配置
    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,  # step 到了 10000 的时候会执行 validation_step；训练的时候
        check_val_every_n_epoch=check_val_every_n_epoch,  # 因为val_check_interval 不为 None，epoch 没有用，因此这个需要是 None
        default_root_dir=None,
        devices=gpus,
        # gradient_clip_val=config.training.gradient_clip_val,  # 1.0
        gradient_clip_algorithm='value',
        # limit_train_batches=config.training.limit_train_batches,  # 1
        # limit_val_batches=config.validation.limit_val_batches,  # 1
        logger=tb_logger,
        # log_every_n_steps=config.logging.train.log_every_n_steps,  # self.logger.log_dict() 每过 log_every_n_steps 次记录一次
        plugins=None,
        precision=16,  # 直接半精度训练
        max_epochs=100,
        max_steps=30000,
        strategy=strategy,  # DDPStrategy  # pytorch_lightning 2.0.6 不知怎的报错, 所以我退回到了 1.8.6
        sync_batchnorm=False if strategy is None else True,
    )

    trainer.fit(model=net_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
