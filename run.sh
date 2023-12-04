#python main.py --alpha 0.2 --target_update_interval 10000 --lr 1e-6 --exp-case case2 --cuda
tensorboard --logdir=runs --host localhost --port 8088
python main.py --automatic_entropy_tuning True --target_update_interval 1000 --lr 1e-4 --exp-case case4 --cuda
#python main.py --alpha 0.2 --target_update_interval 1000 --lr 1e-4 --exp-case case4 --cuda