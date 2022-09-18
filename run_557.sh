echo "seed 557 train"
python T5.py --except_domain train
python T5.py --model_checkpoint t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_train_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.1 --mode finetune

echo "seed 557 restaurant"
python T5.py --except_domain restaurant
python T5.py --model_checkpoint t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_restaurant_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.1 --mode finetune

echo "seed 557 attraction"
python T5.py --except_domain attraction
python T5.py --model_checkpoint t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_attraction_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.1 --mode finetune

echo "seed 557 taxi"
python T5.py --except_domain taxi
python T5.py --model_checkpoint t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.01 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.05 --mode finetune
python T5.py --model_checkpoint t5-smallt5_except_domain_taxi_slotlang_slottype_lr_0.0001_epoch_1_seed_557 --n_epochs 15 --fewshot 0.1 --mode finetune