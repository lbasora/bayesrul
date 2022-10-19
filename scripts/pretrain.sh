#poetry run python -m bayesrul.tasks.train experiment=ncmapss_nn trainer.max_epochs=1,2,3,4,5,6,7,8,9,10 -m
#poetry run python -m bayesrul.tasks.train experiment=ncmapss_nn trainer.max_epochs=1 #--cfg job
poetry run python -m bayesrul.tasks.train experiment=ncmapss_lrt
