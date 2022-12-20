poetry run python -m bayesrul.tasks.train experiment=ncmapss_rad seed=1,2,3,4,5 model.mc_samples_eval=20 trainer.deterministic=False task_name=train_rad trainer.devices=[2] --multirun
