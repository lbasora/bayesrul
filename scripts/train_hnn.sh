poetry run python -m bayesrul.tasks.train experiment=ncmapss_hnn seed=6,7,8,9,10 trainer.deterministic=False task_name=train_hnn trainer.devices=[2] --multirun
