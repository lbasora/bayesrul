#poetry run python -m bayesrul.tasks.train experiment=ncmapss_lrt seed=1,2 model.mc_samples_eval=10 trainer.max_epochs=1 +trainer.limit_train_batches=0.05 +trainer.limit_val_batches=0.05 test=True task_name=train_lrt trainer.devices=[2] --multirun
poetry run python -m bayesrul.tasks.train experiment=ncmapss_lrt seed=1,2,3,4,5 model.mc_samples_eval=20 trainer.deterministic=False task_name=train_lrt trainer.devices=[0] --multirun
