echo "seed 557 train"
python T5.py --except_domain train --n_epochs 2 --train_batch_size 8
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune --except_domain train
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune --except_domain train
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune  --except_domain train

echo "seed 557 taxi"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain taxi --n_epochs 2 --train_batch_size 8
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune --except_domain taxi
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune --except_domain taxi
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune --except_domain taxi