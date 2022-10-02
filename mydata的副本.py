import os, random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
import json

import random
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict
from config import get_args
from torch.utils.data import DataLoader, TensorDataset, Dataset
# from finetune_word_static import fine_tune_data_static
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

random.seed(577)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024

def finetune_data(args, path_name):
    target_data=[]
    with open(path_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            # if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
            #         continue
            # if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            # (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
            if args["except_domain"] != "none" and args["except_domain"] in dial_dict["domains"]:
                target_data.append(dial_dict)
                # print('dial_dict["domains"]',dial_dict["domains"])

            random.Random(args["seed"]).shuffle(dials)
            target_data = target_data[:int(len(dials)*args["fewshot"])]   
    return target_data

def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    domain_list=['hotel','attraction','restaurant','taxi','train']
    print(("Reading all files from {}".format(path_name)))
    data = []
    my_data=[]
    pretrain_data=[]
    detail_sentence=[]
    prompt_data=[]
    domain_counter = {}
    # read files
    print('slot',SLOTS)
   
    with open(path_name) as f:
        dials = json.load(f)
        print('length_dials',len(dials))

        data_dict = json_load('./domain_attribute_data.json')
        if dataset != "test" and args["fewshot"] > 0:
            dials = finetune_data(args, path_name)

        if args["fewshot"] > 0:
            if not args['base']:
                data_dict=json_load('domain_attribute_finetune_data_hotel.json')
        if args['full_attribute']:
            data_dict = json_load('./domain_attribute_data.json')

        for dial_dict in dials:
            dialog_memory=""
            dialog_history = ""
            my_dialog_history=""
            per_turn_state=[]
            context=[]
            # print('dial_dict',dial_dict)
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
            (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue
   
            # domain_list=dial_dict["domains"]
            
            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti
                per_turn_state.append({})
                if not args['use_2.4']:

                    # print('ti',ti)
                    for key in turn['state']['slot_values'].keys():
                        if ti == 0:
                            per_turn_state[ti][key] = turn['state']['slot_values'][key]
                        else:
                            if key not in pre_state.keys() or (key in pre_state.keys() and pre_state[key] != turn['state']['slot_values'][key]):
                                per_turn_state[ti][key] = turn['state']['slot_values'][key]
                else:
                    per_turn_state[ti]=turn['state']['turn_label']
                # print('per_turn_state',per_turn_state[ti])
                # accumulate dialogue utterances

                turn_domain=domain_extract(per_turn_state[ti])
                # sentence=prompt_struct(data_dict,slot_temp,description,dataset,tokenizer,turn_domain)

                dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])
                update_input=dialog_memory+f" {tokenizer.sep_token}"+(" System: " + turn["system"] + " User: " + turn["user"])
                my_dialog_history +=  (" System: " + turn["system"] + " User: " + turn["user"])
                my_dialog_history +=(f" {tokenizer.sep_token} ")
                each_turn= (" System: " + turn["system"] + " User: " + turn["user"])
                context.append((" System: " + turn["system"] + " User: " + turn["user"]))

                if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"],SLOTS)
                else:
                    if args['new_update']:
                       slot_values = per_turn_state[ti]
                    else:
                       slot_values = turn["state"]["slot_values"]
                    accumulative_state=turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args['fewshot']>0:
                        if args["except_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                            slot_values = OrderedDict(
                                [(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                            accumulative_state = OrderedDict(
                                [(k, v) for k, v in accumulative_state.items() if args["except_domain"] in k])
                        elif args["only_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                            slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                            accumulative_state = OrderedDict(
                                [(k, v) for k, v in accumulative_state.items() if args["only_domain"] in k])
                    else:
                        if args["except_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                            slot_values = OrderedDict(
                                [(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                            accumulative_state = OrderedDict(
                                [(k, v) for k, v in accumulative_state.items() if args["except_domain"] not in k])
                        elif args["only_domain"] != "none":
                            slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                            slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                            accumulative_state = OrderedDict([(k, v) for k, v in accumulative_state.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_values = OrderedDict(
                            [(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                        accumulative_state = OrderedDict(
                            [(k, v) for k, v in accumulative_state.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                        accumulative_state = OrderedDict([(k, v) for k, v in accumulative_state.items() if args["only_domain"] in k])

                

                turn_belief_list = [str(k)+'-'+str(v) for k,v in accumulative_state.items()]

                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                if "gpt" in args["model_name"]:
                    turn_slots = []
                    turn_slot_values = []
                    if len(dialog_history.split())>800:
                        continue
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset !="test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}" + " " + tokenizer.bos_token
                        
                        output_text = input_text+ " " + turn["state"]["slot_values"].get(slot, 'none').strip() + " " + tokenizer.eos_token
                        slot_text = slot
                        value_text = turn["state"]["slot_values"].get(slot, 'none').strip()

                        data_detail = {
                            "ID":dial_dict["dial_id"],
                            "domains":dial_dict["domains"],
                            "turn_id":turn_id,
                            "dialog_history":dialog_history,
                            "turn_belief":turn_belief_list,
                            "intput_text":input_text,
                            "output_text":output_text,
                            "slot_text":slot_text,
                            "value_text":value_text
                            }
                        data.append(data_detail)

                else:
                    if args['use_slots_data']:
                        for slot in slot_temp:

                            # skip unrelevant slots for out of domain setting
                            if args["except_domain"] != "none" and dataset !="test":
                                if slot.split("-")[0] not in dial_dict["domains"]:
                                    continue
                            if args["slot_lang"] == "human":
                                slot_lang = description[slot]["description_human"]

                            elif args["slot_lang"] == "naive":
                                slot_lang = description[slot]["naive"]

                            elif args["slot_lang"] == "value":
                                slot_lang = description[slot]["naive"]

                            elif args["slot_lang"] == "question":
                                slot_lang = description[slot]["question"]

                            elif args["slot_lang"] == "slottype":
                                slot_lang = description[slot]["slottype"]
                            # print('id',dial_dict['dial_id'])
                            # print("dialog_history", dialog_history)
                            # if len(tokenizer(dialog_history, return_tensors="pt").input_ids[0])>510:
                            #     print('token_id',len(tokenizer(dialog_history, return_tensors="pt").input_ids[0]))
                            # print('state',turn["state"]["slot_values"])
                            # print('slot_values',slot_values)
                            # print('slot',slot)
                            output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                            # print('output_text',output_text)
                            slot_text = slot
                            value_text = slot_values.get(slot, 'none').strip()
                            if args["slot_lang"]=='none':
                               sentence=word_prompt_single(data_dict,slot,args,dataset)
                            else:
                                sentence = word_prompt_single(data_dict, slot, dataset,args,slot_lang)
                            update_prompt_text=update_input+f" {tokenizer.sep_token}" + sentence
                            prompt_text= dialog_history+ f" {tokenizer.sep_token}" + sentence




                            if args['base']:
                                if args["slot_lang"]=="human":
                                    # slot_lang = description[slot]["description_human"]
                                    input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    my_input_text= my_dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    update_base_text=update_input+ f" {tokenizer.sep_token} {slot_lang}?"
                                elif args["slot_lang"]=="naive":
                                    # slot_lang = description[slot]["naive"]
                                    input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    my_input_text = my_dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    update_base_text = update_input + f" {tokenizer.sep_token} {slot_lang}?"
                                elif args["slot_lang"]=="value":
                                    # slot_lang = description[slot]["naive"]
                                    input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                                    my_input_text = my_dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    update_base_text = update_input + f" {tokenizer.sep_token} {slot_lang}?"
                                elif args["slot_lang"]=="question":
                                    slot_lang = description[slot]["question"]
                                    input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                                    my_input_text = my_dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    update_base_text = update_input + f" {tokenizer.sep_token} {slot_lang}?"
                                elif args["slot_lang"]=="slottype":
                                    # slot_lang = description[slot]["slottype"]
                                    input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    my_input_text = my_dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                                    update_base_text = update_input + f" {tokenizer.sep_token} {slot_lang}?"
                                else:
                                    input_text = dialog_history + f" {tokenizer.sep_token} {slot}"
                                    my_input_text = my_dialog_history + f" {tokenizer.sep_token} {slot}?"
                                    update_base_text = update_input + f" {tokenizer.sep_token} {slot}?"
                                if args['new_update']:
                                    data_detail = {
                                        "ID": dial_dict["dial_id"],
                                        "domains": dial_dict["domains"],
                                        "turn_id": turn_id,
                                        "dialog_history": dialog_history,
                                        "turn_belief": turn_belief_list,
                                        # "intput_text":input_text,
                                        "intput_text": update_base_text,
                                        "output_text": output_text,
                                        "slot_text": slot_text,
                                        "value_text": value_text,
                                        "value_list": description[slot]["values"]
                                    }
                                else:

                                    data_detail = {
                                        "ID":dial_dict["dial_id"],
                                        "domains":dial_dict["domains"],
                                        "turn_id":turn_id,
                                        "dialog_history":dialog_history,
                                        "turn_belief":turn_belief_list,
                                        # "intput_text":input_text,
                                        "intput_text":input_text,
                                        "output_text":output_text,
                                        "slot_text":slot_text,
                                        "value_text":value_text,
                                        "value_list":description[slot]["values"]
                                        }
                            if args['new_update']:
                                prompt_detail = {
                                    "ID": dial_dict["dial_id"],
                                    "domains": dial_dict["domains"],
                                    "turn_id": turn_id,
                                    "dialog_history": dialog_history,
                                    "turn_belief": turn_belief_list,
                                    "intput_text": update_prompt_text,
                                    "output_text": output_text,
                                    "slot_text": slot_text,
                                    "value_text": value_text,
                                    "value_list": description[slot]["values"]
                                }
                            else:
                                prompt_detail={
                                    "ID":dial_dict["dial_id"],
                                    "domains":dial_dict["domains"],
                                    "turn_id":turn_id,
                                    "dialog_history":dialog_history,
                                    "turn_belief":turn_belief_list,
                                    "intput_text":prompt_text,
                                    "output_text":output_text,
                                    "slot_text":slot_text,
                                    "value_text":value_text,
                                    "value_list":description[slot]["values"]
                                    }

                        
                            # }
                            pre_state=turn['state']['slot_values']
                            # print('my_detail',my_detail)
                            # if my_detail['output_text']!='none [eos]':
                            #     print('my_detail', my_detail)
                            # print('pre_state',pre_state.keys())
                            if args['base']:
                              data.append(data_detail)
                            # my_data.append(my_detail)
                            # pretrain_data.append(pretrain_detail)
                            # detail_sentence.append(pretrain_detail_sentence)
                            prompt_data.append(prompt_detail)
                            # print('data',data_detail)
                    else:
                        
                        
                        for domain in domain_list:
                            # print('domain',domain,dataset)
                            if args["except_domain"] != "none":
                               if dataset!='test' and domain in  args["except_domain"]:
                                   continue
                            domain_slot_list=[]
                            prompt_domain_slot=[]
                            for domain_slot in SLOTS:
                                slot_domain = domain_slot.split('-')[0]
                                if slot_domain==domain:
                                    prompt_domain_slot.append([domain_slot,slot_values.get(domain_slot, 'none').strip()])
                            for domain_slot in slot_temp:

                                slot_domain=domain_slot.split('-')[0]
                                # print('slot',slot,slot_domain)
                                # skip unrelevant slots for out of domain setting
                                if args["except_domain"] != "none" and dataset !="test":
                                    if domain_slot.split("-")[0] not in dial_dict["domains"]:
                                        continue
                                # print("dialog_history",dialog_history)
                                # print('slot_values',slot_values)
                                # print('slot',slot)
                                # output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                                # print('output_text',output_text)
                                slot_text = domain_slot
                                value_text = slot_values.get(domain_slot, 'none').strip()
                                domain_slot_list.append([domain_slot,value_text])
                            if dataset=='test':
                                # print('domain',domain)
                                if domain not in args['except_domain']:
                                    continue
                            # print('not_continue')
                            # print('prompt_domian_slot',domain,prompt_domian_slot)
                            sentence=domain_prompt_struct_single(data_dict,domain,dataset,args)
                            prompt_output=domain_data_target(prompt_domain_slot,tokenizer,label=False)
                            # print('prompt_output',prompt_output)
                            # print('sent',sentence)
                            prompt_text= dialog_history+ f" {tokenizer.sep_token}" + sentence
                            update_prompt_text=update_input+f" {tokenizer.sep_token}" + sentence+f" {tokenizer.sep_token}"+prompt_output
                            if len(domain_slot_list)==0:
                                continue
                            # print('domain_list',dial_dict["domains"],domain,domain_slot_list)
                            # print('domain_slot_list',domain_slot_list)
                            domain_output=domain_data_target(prompt_domain_slot,tokenizer)
                            # print(" domain_output,domain_output_label",domain_output,domain_slot_list)
                            prompt_detail={
                                "ID":dial_dict["dial_id"],
                                "domains":dial_dict["domains"],
                                "turn_id":turn_id,
                                "dialog_history":dialog_history,
                                "turn_belief":turn_belief_list,
                                "intput_text":update_prompt_text,
                                "output_text":domain_output,
                                "domain":domain,
                                "value_text":prompt_domain_slot
                                }

                        
                            # print('my_detail',my_detail)
                            # if my_detail['output_text']!='none [eos]':
                            #     print('my_detail', my_detail)
                            # print('pre_state',pre_state.keys())
                            
                            # my_data.append(my_detail)
                            # pretrain_data.append(pretrain_detail)
                            # detail_sentence.append(pretrain_detail_sentence)
                            prompt_data.append(prompt_detail)
                if not args['use_2.4']:
                  pre_state=turn['state']['slot_values']
                dialog_memory+=(" System: " + turn["system"] + " User: " + turn["user"])
    # {'ID': 'SNG01856.json', 'domains': ['hotel'], 'turn_id': 0, 'dialog_history': ' System: none User: am looking for a place to to stay that has cheap price range it should be in a type of hotel',
    # 'turn_belief': ['hotel-pricerange-cheap', 'hotel-type-hotel'], 'intput_text': ' System: none User: am looking for a place to to stay that has cheap price range it should be in a type of hotel [sep] price budget of the hotel?',
    # 'output_text': 'cheap [eos]', 'slot_text': 'hotel-pricerange', 'value_text': 'cheap', 'value_list': ['cheap', 'dontcare', 'expensive', 'moderate']}
    # print(len(data))
    # print('len',len(prompt_data))
    # print("domain_counter", domain_counter)
    # if dataset=='train':
    #     for detail in prompt_data :
    #         print('ID',detail['ID'])
    #         print('input',detail['intput_text'])
    #         print("value_text",detail["turn_belief"])
    #         print("output_text",detail['output_text'])
            
    if args['base']==True:
       return data, slot_temp
    else:
       return prompt_data, slot_temp
def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS
def domain_extract(state_list):
    domain_list=[]
    for slot in state_list:
        end_symbol='-'
        symbol_idx=slot.find(end_symbol)
        domain=slot[:symbol_idx]
        domain_list.append(domain)
    if len(domain_list)!=0:      
        domain_list=list(set(domain_list))
    return domain_list
    # print('domain_list',domain_list)
    # if len(domain_list)>1:
    #    print('domain_list2',domain_list) 
class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)


def gpt_collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                             return_attention_mask=False, truncation=True, max_length=1000)
    batch_data["input_ids"] = output_batch['input_ids']
    return batch_data


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                            verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                             return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']
    # print('batch_data["decoder_output"]',batch_data["decoder_output"])
    return batch_data
def prepare_data(args, tokenizer):
    if args['use_2.4']:
        path_train = 'MultiWOZ2.4-main/data/mwz2.4/train_dials_translate.json'
        path_dev = 'MultiWOZ2.4-main/data/mwz2.4/dev_dials_translate.json'
        path_test = 'MultiWOZ2.4-main/data/mwz2.4/test_dials_translate.json'

        ontology = json.load(open("MultiWOZ2.4-main/data/mwz2.4/ontology.json", 'r'))
    elif args['use_2.0']:
        path_train = 'data_2.0/data/train_dials.json'
        path_dev = 'data_2.0/data/dev_dials.json'
        path_test = 'data_2.0/data/test_dials.json'
        ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    else:
        path_train = 'data/train_dials.json'
        path_dev = 'data/dev_dials.json'
        path_test = 'data/test_dials.json'

        ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))

    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("utils/slot_description.json", 'r'))

    data_train, _ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train")
    data_dev, _ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")


    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    if "gpt" in args["model_name"]:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    fewshot_loader_dev=None
    fewshot_loader_test=None
    return train_loader, dev_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test
def dsiturb_context(context,per_turn_state,slot,tokenizer,slot_lang):
    context_data=[]
    save_idx=-1
    answer_idx=-1
    for idx,data in enumerate(context):
        context_data.append([data,per_turn_state[idx]])
        if slot in per_turn_state[idx].keys():
            if save_idx==-1:
               save_idx=idx
            else:
               context_data[save_idx]='del_token'
               save_idx = idx
    if save_idx!=-1:
        pretrain_output=context_data[save_idx][0]
        value_text=context_data[save_idx][0]
    else:
        pretrain_output='none [eos]'
        value_text='none'
    random.shuffle(context_data)
    my_history=''
    idx=0
    for data in context_data:
        if data!='del_token':
          my_history=my_history+str(idx)+':'+data[0]+f" {tokenizer.sep_token}"
          if pretrain_output==data[0]:
              answer_idx=idx
          idx+=1
        else:
            continue
    if pretrain_output=='none [eos]':
        answer_idx = idx
    my_history = my_history+str(idx)+':'+'none'+f" {tokenizer.sep_token}"
    my_history=my_history+slot_lang
    # my_history = my_history + 'desc:'+slot_lang+'?'
    my_history='[pre]'+' '+my_history
    return my_history,pretrain_output,value_text,str(answer_idx)+f" {tokenizer.eos_token}",str(answer_idx)
def dsiturb_context_sentence(context,per_turn_state,slot,tokenizer,slot_lang):
    context_data=[]
    save_idx=-1
    answer_idx=-1
    for idx,data in enumerate(context):
        context_data.append([data,per_turn_state[idx]])
        if slot in per_turn_state[idx].keys():
            if save_idx==-1:
               save_idx=idx
            else:
               context_data[save_idx]='del_token'
               save_idx = idx
    context_data.append(['none',''])
    if save_idx!=-1:
        pretrain_output=context_data[save_idx][0]
        value_text=context_data[save_idx][0]
    else:
        pretrain_output='none'
        value_text='none'
    random.shuffle(context_data)
    my_history=''
    for data in context_data:
        if data!='del_token':
          my_history=my_history+data[0]+f" {tokenizer.sep_token}"        
        else:
            continue
    # my_history = my_history+'none'+f" {tokenizer.sep_token}"
    my_history=my_history+slot_lang+'?'
    # my_history = my_history + 'desc:'+slot_lang+'?'
    my_history='[pre]'+' '+my_history
    return my_history,pretrain_output+f" {tokenizer.eos_token}",pretrain_output

def json_load(path):
    with open(path) as json_data:
      data = json.load(json_data)
    return data
def prompt_struct(data_dict,slot_list,description,dataset,tokenizer,domain_list=None):
    
    domain_slot=[]
    for domain in domain_list:
        for slot in slot_list:
            if domain in slot:
               domain_slot.append(slot)
    # print(len(domain_slot),domain_slot)
    
    sentence=''
    sentence+='[STATE_OUT]'
    sentence+=' '
    for slot in domain_slot:
        
        sentence+='<slot_name>'
        sentence+=' '
        prompt=list_append(slot,data_dict[dataset][slot])
        sentence+=prompt
        sentence+=' '
        sentence+='<slot_name\>'
        sentence+=' '
        # print('sentence_id',tokenizer.convert_tokens_to_ids('eat an apple'))
    # print('sentence',sentence)
    # if (len(tokenizer(sentence, return_tensors="pt").input_ids[0]))>400:
    #    print('sentence',sentence)
    #    print('sentence_id',len(tokenizer(sentence, return_tensors="pt").input_ids[0]))
    
    return sentence
def random_slot_token(token_list,attribute,attribute_dict,slot,args):
    if args['noise']:
       random_num=random.randint(1, 10)
    else:
        random_num=random.randint(2, args['max_num'])
    if 3<=random_num and args['max_num']:
        random.shuffle(token_list)
        if len(token_list)!=0:
            idx=random.randint(0, len(token_list))
            result=token_list[:idx]
        else:
            result=[]

        # print('random',token_list,result)
    if 2==random_num :
        result=[]

        # print('empty')
    if  random_num==1:
         noise_idx=random.randint(1, 2)
         if noise_idx==1:
             slot_list =  slot_filter(slot,list(attribute_dict.keys()))

             slot_idx=random.randint(0,len(slot_list)-1)
             result=attribute_dict[slot_list[slot_idx]][attribute]
         else:
             slot_list = slot_filter(slot, list(attribute_dict.keys()))

             slot_idx = random.randint(0, len(slot_list) - 1)
             result = attribute_dict[slot_list[slot_idx]][attribute]
             if len(result)!=0:
                 result=result[0:random.randint(0, len(result)-1)]
        #  print('noise',token_list,result)

    return result
def slot_filter(slot,slot_list):
    result=[]
    slot_domain=slot.split('-')[0]

    for data in slot_list:
        if data.split('-')[0]==slot_domain:
            if data!=slot:
               result.append(data)
    return result
def word_prompt_generate(data_dict,dataset,all_dict,slot,args):
    sentence=' , which may include '
    attributes = list(data_dict.keys())
    str_n=''
    str_v=''
    str_output=''
    for attribute in attributes:
        if attribute == 'slot_n':
            if dataset=='train':
                random_list=random_slot_token(data_dict[attribute],attribute,all_dict,slot,args)
                for idx,word in enumerate(random_list):
                    str_n+=word
                    if idx!=len(random_list)-1:
                       str_n+= ','
            else:
                for idx, word in enumerate(data_dict[attribute]):
                    str_n += word
                    if idx != len(data_dict[attribute]) - 1:
                        str_n += ','
        if attribute == 'slot_v':
            if dataset == 'train':
                random_list= random_slot_token(data_dict[attribute],attribute,all_dict,slot,args)
                for idx,word in enumerate(random_list ):
                    str_v += word
                    if idx!=len(random_list )-1:
                       str_v += ','
            else:
                for idx, word in enumerate(data_dict[attribute]):
                    str_v += word
                    if idx != len(data_dict[attribute]) - 1:
                        str_v += ','
        if attribute == 'output_type':
            str_output+=data_dict[attribute][0]
    str_n='{ '+str_n+" }"
    str_v='{ '+str_v+" }"
    str_output='{ '+str_output+" }"
    sentence=sentence+f"{str_n}"+' or '+f"{str_v}"+', '+'its output type is '+str_output

    return sentence
def word_prompt_single(data_dict,slot,dataset,args,slot_lang=None):
    # if args["fewshot"]==0:
    # if dataset == 'test':
    #     dataset = 'train'
    if slot_lang == None:
        sentence=slot
    else:
        sentence=slot_lang
    sentence+=word_prompt_generate(data_dict[dataset][slot],dataset,data_dict[dataset],slot,args)
    return sentence
def prompt_struct_single(data_dict,slot,dataset,args,slot_lang=None):
    # if args["fewshot"]==0:
    if dataset=='test':
        dataset='train'
    sentence=''
    sentence+='[STATE_OUT]'
    sentence+=' '
    sentence+='<slot_name>'
    sentence+=' '
    if slot_lang==None:
       prompt=list_append(slot,data_dict[dataset][slot])
    else:
      prompt = list_append(slot_lang, data_dict[dataset][slot])
    sentence+=prompt
    sentence+=' '
    sentence+='<slot_name\>'
    sentence+=' '
        # print('sentence_id',tokenizer.convert_tokens_to_ids('eat an apple'))
    
    # if (len(tokenizer(sentence, return_tensors="pt").input_ids[0]))>200:
    #    print('sentence',sentence)
    #    print('sentence_id',len(tokenizer(sentence, return_tensors="pt").input_ids[0]))
    
    return sentence
def domain_prompt_struct_single(data_dict,domain,dataset,args):
    if args['except_domain']=="none":
        if dataset=='test':
            dataset='train'
    if args['mode']=='finetune':
       dataset='test'
    sentence=''
    sentence+='[STATE_OUT]'
    sentence+=' '
    sentence+='<domain_name>'
    sentence+=' '
    prompt=list_append(domain,data_dict[dataset][domain])
    sentence+=prompt
    sentence+=' '
    sentence+='<domain_name\>'
    sentence+=' '
        # print('sentence_id',tokenizer.convert_tokens_to_ids('eat an apple'))
    
    # if (len(tokenizer(sentence, return_tensors="pt").input_ids[0]))>200:
    #    print('sentence',sentence)
    #    print('sentence_id',len(tokenizer(sentence, return_tensors="pt").input_ids[0]))
    
    return sentence
def list_append(slot,prompt_dict):
    result=''
    result+=slot
    result+=':'
    result+=' '
    result+='{'
    result+=' '
    attributes=list(prompt_dict.keys())
    for attribute in attributes:
        if attribute=='slot_n':
            perfix='<slot_n>'
            suffix='<slot_n\>'
        if attribute=='slot_v':
            # continue
            perfix='<slot_v>'
            suffix='<slot_v\>'
        if attribute=='value':
            perfix='<value>'
            suffix='<value\>'

        if attribute=='output_type':
            perfix='<output_type>'
            suffix='<output_type\>'
        if attribute=='slot':
            perfix='<slot_name>'
            suffix='<slot_name\>'
        
        result+=perfix
        result+=' '
        for value in prompt_dict[attribute]:
            result+=value
            result+=' '
            
        result+=suffix
        result+=' '
    result+='}'
    return result
def domain_data_target(data,tokenizer,label=True):
    if label:
        target=""
        for idx,slot_value in enumerate(data):
            if idx!=len(data)-1:
                target=target+domain_slot(slot_value[0])+' '+'is'+' '+slot_value[1]+','
            else:
                 target=target+domain_slot(slot_value[0])+' '+'is'+' '+slot_value[1]
        target+=f" {tokenizer.eos_token}"
    else:
        target = ""
        for idx, slot_value in enumerate(data):
            if idx != len(data) - 1:
                target = target + domain_slot(slot_value[0]) + ' ' + 'is' + ' ' + f" {tokenizer.mask_token}" + ','
            else:
                target = target + domain_slot(slot_value[0]) + ' ' + 'is' + ' ' + f" {tokenizer.mask_token}"
        target += f" {tokenizer.eos_token}"

    return target 
def domain_slot(data):
    data=data.split('-')
    result='the'+' '+data[1]+' '+'of'+' '+data[0]
    return result

if __name__ == "__main__":
    from tqdm import tqdm
    args = get_args()
    args = vars(args)
    # tokenizer = T5Tokenizer.from_pretrained('./save/pre_T5t5-smallt5_except_domain_hotel_slotlang_human_lr_0.0001_epoch_1_seed_590/')
    # tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]",mask_token="[mask]")
    print(tokenizer.mask_token)
    print(args["model_checkpoint"])
    # print('sentence_id',tokenizer.convert_tokens_to_ids('eat an apple'))
    # tokenizer.add_special_tokens({'additional_special_tokens': ["[pre]"]})
    # a=' [sep][STATE_OUT] <domain_name> restaurant: { <slot_name> food pricerange area name book time book day book people <slot_name\> <slot_v> find have looking book <slot_v\> <slot_n> restaurant food table centre town price range people area place number <slot_n\> } <domain_name\> '
    # print(tokenizer(a, return_tensors="pt").input_ids,len(tokenizer(a, return_tensors="pt").input_ids[0]))
    special_tokens=['[TYPE_OUT]','<slot_v>','<slot_v\>','<slot_n>','<slot_n\>','<value_adv>','[STATE_OUT]','<slot_name>','<slot_name\>','<value>','<value\>','{','}','<slot_name>','<slot_name\>']
    slot_name= ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day', 'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-book people', 'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'attraction-name', 'restaurant-name', 'attraction-type', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure', 'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arriveby']
    tokenizer.add_tokens(special_tokens)
    tokenizer.add_tokens(slot_name)
    # print(tokenizer(a, return_tensors="pt").input_ids,len(tokenizer(a, return_tensors="pt").input_ids[0]))
    args['slot_lang'] = 'naive'
    args["except_domain"] = 'hotel'
    # args['fewshot']=0.01
    args['use_slots_data']=True
    args['new_update']=True
    # args['use_2.4'] = True
    # args['use_domain_slots_data']=True
    # args['base'] = True
    # print(tokenizer.convert_tokens_to_ids('[pre]'))
    # print(tokenizer.additional_special_tokens)
    # print(tokenizer.additional_special_tokens_ids)
    normal_str='the destination of train is none,the day of train is none,the departure of train is none,the arriveby of train is none,the book people of train is none,the leaveat of train is none,the food of restaurant is none,the pricerange of restaurant is none,the area of restaurant is centre,the name of restaurant is none,the book time of restaurant is none,the book day of restaurant is none,the book people of restaurant is none [eos]'
    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args,tokenizer)
    # for i in range(10):
    #   print(random.randint(0, 5))
    # print(("restaurant-food is french").split(' is '))
    # state='hotel-book day-tuesday'
    # one_str='the destination of train is none [sep]the day of train is none [sep]the departure of train is none [sep]the arriveby of train is none [sep]the book people of train is none [sep]the leaveat of train is none [sep]the area of attraction is none [sep]the name of attraction is the man on the moon [sep]the type of attraction is none [eos]'
    # second_str='the destination of train is none the day of train is saturday the departure of train is none the arriveby of train is 10.30 the book people of train is none the leaveat of train is none'
    # third_str='the destination of train is none'
    # next_str='day of train is saturday'
    # error_str='the area of attraction is none the name of attraction is none the type of attraction is none the leaveat of taxi is none the destination of taxi is the junction the departure of taxi is express by holiday inn cambridge the arriveby of taxi is none'
    # print('split',normal_str.split(','))
    # print('split',second_str.split('the '))
    # print('split',next_str.split(' of ')[0].split(' ')[0])
    # print('split',error_str.split('the '))
    # print('sentence_id',tokenizer.convert_tokens_to_ids(''))
    # print(tokenizer('the destination of train is none [sep]the day of train is none [sep]the departure of train is none [sep]the arriveby of train is none [sep]the book people of train is none [sep]the leaveat of train is none [eos]', return_tensors="pt").input_ids)
   
    
    # print('sentence',tokenizer.convert_ids_to_tokens([ 1175,    44, 32101]))
    # for batch in tqdm(test_loader):
    #     print('batch',batch)

  