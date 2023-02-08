#poetry run python -m bayesrul.tasks.train task_name=test_lrt datamodule.batch_size=10000 experiment=ncmapss_lrt train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/LRT/0/checkpoints/epoch_111-step_266784.ckpt
#poetry run python -m bayesrul.tasks.train task_name=test_mcd datamodule.batch_size=10000 experiment=ncmapss_mcd train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/MCD/0/checkpoints/epoch_204-step_488310.ckpt
#poetry run python -m bayesrul.tasks.train task_name=test_hnn datamodule.batch_size=10000 experiment=ncmapss_hnn train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/HNN/0/checkpoints/epoch_326-step_311631.ckpt

# poetry run python -m bayesrul.tasks.train task_name=test_mcd0 datamodule.batch_size=10000 experiment=ncmapss_mcd train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/MCD/0/checkpoints/epoch_204-step_488310.ckpt
# poetry run python -m bayesrul.tasks.train task_name=test_mcd1 datamodule.batch_size=10000 experiment=ncmapss_mcd train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/MCD/1/checkpoints/epoch_173-step_414468.ckpt
# poetry run python -m bayesrul.tasks.train task_name=test_mcd2 datamodule.batch_size=10000 experiment=ncmapss_mcd train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/MCD/2/checkpoints/epoch_248-step_593118.ckpt
# poetry run python -m bayesrul.tasks.train task_name=test_mcd3 datamodule.batch_size=10000 experiment=ncmapss_mcd train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/MCD/3/checkpoints/epoch_299-step_714600.ckpt
# poetry run python -m bayesrul.tasks.train task_name=test_mcd4 datamodule.batch_size=10000 experiment=ncmapss_mcd train=False test=True  ckpt_path=/home/luis/repos/bayesrul/results/ncmapss/runs/MCD/4/checkpoints/epoch_281-step_671724.ckpt

poetry run python -m bayesrul.tasks.train task_name=test_fo datamodule.batch_size=10000 experiment=ncmapss_fo train=False test=True ckpt_path=/home/luis/repos/bayesrul/logs/train_fo/multiruns/2022-11-01_10-11-05/3/checkpoints/epoch_137-step_328716.ckpt