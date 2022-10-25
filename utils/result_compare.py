import json
import os.path


def json_load(path):
    with open(path) as json_data:
      data = json.load(json_data)
    return data
def json_save(save_path,result):
   with open(save_path, 'w') as f:
       json.dump(result, f, indent=4)
def compare(path1,path2,save_path,compare_with_other=False):
   result={}
   if compare_with_other == True:
      base_result=json_load(path1)
   compare_result=json_load(path2)
   name_list=list(compare_result.keys())
   for name in name_list:
      result[name]={}
      com_turn=compare_result[name]['turns']
      if compare_with_other == True:
          base_turn = base_result[name]['turns']

      turn_num=list(com_turn.keys())
      for turn in turn_num:
         if compare_with_other == True:
             base_state = base_turn[turn]['pred_belief']
         else:
             base_state=com_turn[turn]['turn_belief']
         com_state=com_turn[turn]['pred_belief']
         turn_state=com_turn[turn]['turn_belief']
         # if base_state!=turn_state or com_state!=turn_state :
         if base_state!=com_state:
               result[name][turn]={}
               result[name][turn]['turn_belief']=turn_state
               result[name][turn]['base_belief'] = base_state
               result[name][turn]['com_belief'] = com_state
   json_save(save_path, result)

if __name__ == "__main__":
   base_path='../save/t5-smallt5_except_domain_taxi_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_0.0_base_True_task2first_False/results/ratio_0.1_seed_558_prediction.json'
   com_path= '../save/t5-smallt5_except_domain_taxi_lr_0.0001_epoch_1_seed_558_batch_size_4_auxiliary_task_1.0_base_False_task2first_True/results/ratio_0.1_seed_558_prediction.json'
   save_path='taxi_compare_2.1.json'
   compare(base_path, com_path, save_path,True)

   # ratio_0.01_seed_595_prediction.json
   # ratio_0.05_seed_595_prediction.json
   # zeroshot_prediction.json