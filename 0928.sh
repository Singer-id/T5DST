#CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain taxi --n_epochs 1 --train_batch_size 8 --seed 558
#CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 5 --seed 558 --fewshot 0.01 --mode finetune
#CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 5 --seed 558 --fewshot 0.05 --mode finetune
#CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 5 --seed 558 --fewshot 0.1 --mode finetune

#base 要加上--base True
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain train --n_epochs 1 --train_batch_size 8 --seed 558
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 5 --seed 558 --fewshot 0.01 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 5 --seed 558 --fewshot 0.05 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 5 --seed 558 --fewshot 0.1 --mode finetune