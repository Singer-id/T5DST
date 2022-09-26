#hotel域消融实验
echo "seed 557 hotel"
python T5.py --except_domain hotel --n_epochs 2 --train_batch_size 8 --auxiliary_task_ratio 0.0
python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_train_batch_size_8_epoch_2_seed_557_auxiliary_task_ratio_0.0/ --n_epochs 5 --fewshot 0.01 --mode finetune
#python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_train_batch_size_8_epoch_2_seed_557_auxiliary_task_ratio_0.0/ --n_epochs 5 --fewshot 0.05 --mode finetune
#python T5.py --model_checkpoint save/t5-smallt5_except_domain_hotel_train_batch_size_8_epoch_2_seed_557_auxiliary_task_ratio_0.0/ --n_epochs 5 --fewshot 0.1 --mode finetune


