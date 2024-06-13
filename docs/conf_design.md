# config 设计逻辑梳理
```bash
config -> main(); fetch_data_module(); fetch_net_module().
    dataset_name: {gen1, gen4, DSEC, NCaltec101, ...}
    dataset_config(不同的数据集有不用的dataset_config) -> DataModule
        num_workers_train: 8
        num_workers_evil: 8
        batch_size_train: 24
        batch_size_eval: 24
    network_config -> NetModule
    
    
    training
        lr_scheduler
            use
    
    validataion
        val_check_interval
        chech_val_every_n_epoch
        
```

## 使用 hydra
- `defaults` 是 hydra 中的一个特殊的 key，其内部



