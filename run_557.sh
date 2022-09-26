echo "seed 557 hotel"
python T5.py --except_domain hotel --n_epochs 2 --train_batch_size 8
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune

echo "seed 557 train"
python T5.py --except_domain train --n_epochs 2 --train_batch_size 8
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune

echo "seed 557 restaurant"
python T5.py --except_domain restaurant --n_epochs 2 --train_batch_size 8
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune

echo "seed 557 attraction"
python T5.py --except_domain attraction --n_epochs 2 --train_batch_size 8
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune

echo "seed 557 taxi"
python T5.py --except_domain taxi --n_epochs 2 --train_batch_size 8
python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint save/t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_2_seed_557 --n_epochs 5 --fewshot 0.1 --mode finetune

#echo "full-shot"
#python T5.py --train_batch_size 8 --n_epochs 5