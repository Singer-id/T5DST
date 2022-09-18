group=2
domian_list=" taxi "
model_check="t5-small"
prompt_save="./prompt_save/"
base_save="./base_save"
model_name='t5'
lr=0.0001
epoch=5
batch_size=4
loop_list=" 4 "
if [ $group = 1 ];then
    echo "prompt4个域"
    for seed in 595;do
        for domain in $domian_list;do
        prompt_path=${prompt_save}${model_check}${model_name}"_except_domain_"${domain}"_lr_"${lr}"_epoch_"${epoch}"_seed_"${seed}
        base_path=${base_save}${model_check}${model_name}"_except_domain_"${domain}"_lr_"${lr}"_epoch_"${epoch}"_seed_"${seed}
        echo ' prompt_path' $prompt_path
        echo 'base_path' $base_path
        echo "seed = " $seed
        echo "prompt" $domain
#        python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang human  --n_epochs 2
        # python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain  --n_epochs 10 --model_checkpoint $prompt_path --fewshot 0.01 --mode finetune --seed $seed
#        python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain --base --seed $seed --new_update --use_slots_data --slot_lang human --n_epochs 2
        # python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain --base --n_epochs 10 --model_checkpoint $base_path --fewshot 0.01 --mode finetune --seed $seed
        python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang human  --n_epochs 2   --use_2.4 --lr 0.5e-4
        done
    done
fi

if [ $group = 2 ];then
   echo "prompt1个域"
    for seed in 595;do
        for domain in $domian_list;do
        prompt_path=${prompt_save}${model_check}${model_name}"_except_domain_"${domain}"_lr_"${lr}"_epoch_"${epoch}"_seed_"${seed}
        base_path=${base_save}${model_check}${model_name}"_except_domain_"${domain}"_lr_"${lr}"_epoch_"${epoch}"_seed_"${seed}
        echo ' prompt_path' $prompt_path
        echo 'base_path' $base_path
        echo "seed = " $seed
        echo "prompt" $domain
        python prompt_T5.py --train_batch_size $batch_size --GPU 2 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang naive  --n_epochs 3  --lr 0.05e-4
        python prompt_T5.py --train_batch_size $batch_size --GPU 2 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang naive  --n_epochs 4  --lr 0.05e-4
        # python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain  --n_epochs 10 --model_checkpoint $prompt_path --fewshot 0.01 --mode finetune --seed $seed
        # python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain --base --seed $seed --new_update --use_slots_data --slot_lang human --n_epochs 2 --use_domain_slots_data
        # python prompt_T5.py --train_batch_size $batch_size --GPU 1 --except_domain $domain --base --n_epochs 10 --model_checkpoint $base_path --fewshot 0.01 --mode finetune --seed $seed
        
        done
    done

fi

if [ $group = 3 ];then
    echo "train finetune "
    for seed in 595;do
      for domain in $domian_list;do
        for loop in $loop_list;do
          if [ $loop=1 ];then
             echo "1"
            python prompt_T5.py --train_batch_size $batch_size --GPU 2 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang naive  --n_epochs 2 --max_num 9 
          fi

          if [ $loop=2 ];then
             echo "2"
            python prompt_T5.py --train_batch_size $batch_size --GPU 2 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang naive  --n_epochs 2 --max_num 11
          fi

          if [ $loop=3 ];then
             echo "3"
             python prompt_T5.py --train_batch_size $batch_size --GPU 2 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang naive  --n_epochs 2 --max_num 13
          fi

          if [ $loop=4 ];then
             echo "4"
            python prompt_T5.py --train_batch_size $batch_size --GPU 2 --except_domain $domain --seed $seed --new_update --use_slots_data --slot_lang naive  --n_epochs 2 --max_num 15
          fi
        done
      done
    done
fi