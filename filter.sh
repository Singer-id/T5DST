#filter
python T5.py --except_domain restaurant --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0  --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

python T5.py --except_domain train --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

python T5.py --except_domain attraction --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0  --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

#655
python T5.py --except_domain restaurant --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0  --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

python T5.py --except_domain train --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

python T5.py --except_domain attraction --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0  --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

python T5.py --except_domain hotel --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0  --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first

python T5.py --except_domain taxi --n_epochs 1 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.01 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.05 --mode finetune --task2first
python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True_filter/ --n_epochs 5 --train_batch_size 4 --seed 558 --auxiliary_task_ratio 1.0 --fewshot 0.1 --mode finetune --task2first