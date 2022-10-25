from word_static import json_load


def list_to_dict(input_list):
    result={}
    for data in input_list:
        slot=data[0]+'-'+data[1]
        value=data[2]
        result[slot]=value
    return result
def result_compare(true_dict,pred_dict):
    surplus=[]
    miss=[]
    error=[]
    true_slot=list(true_dict.keys())
    pred_slot=list(pred_dict.keys())
    for slot in true_slot:
        if slot not in pred_slot:
            miss.append(slot)
    for slot in pred_slot:
        if slot not in true_slot:
            surplus.append(slot)
        else:
            if pred_dict[slot]!=true_dict[slot]:
                error.append(slot)
    return surplus,miss,error

def dict_check(result_dict,input_dict):
    for slot in input_dict.keys():
        if slot not in list(result_dict.keys()):
            result_dict[slot]=[0]
            
def error_ratio(total,result_dict):
    for slot in result_dict.keys():
        result_dict[slot].append(total)
        result_dict[slot].append(float(result_dict[slot][0]/total))
if __name__=="__main__":  
    error_dict={}
    surplus_dict={}
    miss_dict={}
    data_dict=json_load('train_compare_2.0.json')

    name_list=list(data_dict.keys())
    total_surplus=0
    total_miss=0
    total_error=0
    for name in name_list:
        
        log=data_dict[name]
        turn_list=list(log.keys())
        for turn in turn_list:
            data=log[turn]
            # turn_belief=data['turn_belief']
            pred_belief=list_to_dict([x.split('-') for x in data['com_belief']])
            turn_belief=list_to_dict([x.split('-') for x in data['turn_belief']])
            # print('turn_belief',turn_belief)
            dict_check(error_dict, pred_belief)
            dict_check(error_dict,turn_belief)
            dict_check(surplus_dict, pred_belief)
            dict_check(surplus_dict,turn_belief)
            dict_check(miss_dict, pred_belief)
            dict_check(miss_dict,turn_belief)
            surplus,miss,error=result_compare(turn_belief,pred_belief)
            for slot in surplus:
                surplus_dict[slot][0]+=1
                total_surplus+=1
            for slot in miss:
                miss_dict[slot][0]+=1
                total_miss+=1
            for slot in error:
                error_dict[slot][0]+=1
                total_error+=1
    error_ratio(total_surplus,surplus_dict)
    error_ratio(total_miss,miss_dict)
    error_ratio(total_error,error_dict)
    print('surplus',surplus_dict)
    print('miss',miss_dict)
    print('error',error_dict)
           
