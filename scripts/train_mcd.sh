poetry run python -m bayesrul.tasks.train experiment=ncmapss_mcd seed=6,7,8,9,10 model.mc_samples=20 trainer.deterministic=False task_name=train_mcd trainer.devices=[2] --multirun
