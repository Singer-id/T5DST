# base未做
echo "seed 558 hotel"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain hotel --seed 558
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.01 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.05 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.1 --seed 558 --mode finetune

echo "seed 558 train"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain train --seed 558
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.01 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.05 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.1 --seed 558 --mode finetune

echo "seed 558 restaurant"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain restaurant --seed 558
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.01  --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.05 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.1 --seed 558 --mode finetune

echo "seed 558 attraction"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain attraction --seed 558
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.01 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.05 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.1 --seed 558 --mode finetune

echo "seed 558 taxi"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain taxi --seed 558
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.01 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.05 --seed 558 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_558 --n_epochs 15 --fewshot 0.1 --seed 558 --mode finetune

echo "seed 559 hotel"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain hotel --seed 559
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.01 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.05 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.1 --seed 559 --mode finetune

echo "seed 559 train"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain train --seed 559
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.01 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.05 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.1 --seed 559 --mode finetune

echo "seed 559 restaurant"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain restaurant --seed 559
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.01 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.05 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.1 --seed 559 --mode finetune

echo "seed 559 attraction"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain attraction --seed 559
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.01 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.05 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.1 --seed 559 --mode finetune

echo "seed 559 taxi"
CUDA_VISIBLE_DEVICES=1 python T5.py --except_domain taxi --seed 559
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.01 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.05 --seed 559 --mode finetune
CUDA_VISIBLE_DEVICES=1 python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_559 --n_epochs 15 --fewshot 0.1 --seed 559 --mode finetune