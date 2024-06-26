from datasets598.utils import load_ade, load_conll04, load_nyt
from utils import *
import json
import os
from tqdm import tqdm
import argparse
import shutil


def get_response_from_llm(args):
    dataset = args.dataset
    out_dir = os.path.join(f'outputs/{args.suffix}')
    os.makedirs(out_dir, exist_ok=True) 

    # load prompts
    prompt_dir = args.prompt_dir if args.prompt_dir else os.path.join('prompts', dataset)
    prompt_path_entity = os.path.join(prompt_dir, 'prompt_tot_entity.txt')
    with open(prompt_path_entity, 'r') as f:
        prompt_entity = f.read()
    if args.extract_relation_types:
        prompt_path_relation_type = os.path.join(prompt_dir, 'prompt_tot_relation_type.txt')
        with open(prompt_path_relation_type, 'r') as f:
            prompt_relation_type = f.read()
    prompt_path_relation = os.path.join(prompt_dir, 'prompt_tot_relation.txt')
    with open(prompt_path_relation, 'r') as f:
        prompt_relation = f.read()

    # load test data
    if dataset == 'ade':
        data = load_ade(split=args.split)
        test_data = data['test']
        with open(os.path.join(prompt_dir, 'few_shot_id.txt'), 'r') as f:
            for id in f.read().split('\n'):
                test_data.pop(id, None)
    elif dataset == 'conll04':
        data = load_conll04()
        test_data = data['val'] if args.is_val else data['test']
        # get the entity types for each relation type within conll04 dataset
        valid_type_dict = {
            "Work For": set([("Per", "Org")]),
            "Kill": set([("Per", "Per")]),
            "OrgBased In": set([("Org", "Loc")]),
            "Live In": set([("Per", "Loc")]),
            "Located In": set([("Loc", "Loc")])
        }
    elif dataset == 'nyt':
        data = load_nyt()
        test_data = data['val'] if args.is_val else data['test']
    else:
        raise Exception('Dataset Not Supported!')
    if args.test_k >= 0:
        test_data = dict_first_k(test_data, args.test_k)

    # get response; {id: response}
    responses_entity = run_llm(args.api_key, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_entity, test_data)
    if args.extract_relation_types: 
        responses_relation_type = run_llm(args.api_key, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_relation_type, test_data)

    # logic to map the extracted entities / relation types into relations, with the valid type dict
    relation_prompt_string_dict = {} # {id: prompt string for candidate relations}
    for id, response in responses_entity.items():

        # try to parse each response; if fail, pass an empty list
        entity_type_dict = struct_response_entity(response, data['entities'])
        relation_type_list = struct_response_relation(responses_relation_type[id]) if args.extract_relation_types else data['relations']      

        # construct relations; {relation type: [[ins1, int2], [ins3, ins4]]}
        relation_type_dict = {}
        for relation_type in relation_type_list:
            relation_type_dict[relation_type] = []
            # get the all possible entity type pairs
            possible_entity_type_pairs = valid_type_dict[relation_type]
            # for each entity pair, iterate and match
            for entity_type_1, entity_type_2 in possible_entity_type_pairs:
                # get all the instances of both types and match
                entity_instance_list_1 = entity_type_dict[entity_type_1]
                entity_instance_list_2 = entity_type_dict[entity_type_2]
                for instance1 in entity_instance_list_1:
                    for instance2 in entity_instance_list_2:
                        if instance1 == instance2:
                            continue
                        relation_type_dict[relation_type].append([instance1, instance2])
        
        # convert the constructed relations into prompt string
        relation_prompt_string = ''
        for relation, entity_pair_list in relation_type_dict.items():
            relation_prompt_string += '\n' + ('"' + relation + '": ') + str(entity_pair_list)

        # record to the dictionary
        relation_prompt_string_dict[id] = relation_prompt_string
    
    # save the entity output
    with open(os.path.join(out_dir, f'output_{dataset}_seed={args.seed}_split={args.split}_entity.json'), 'w') as json_file:
        json.dump({
            'raw_entity_output': responses_entity,
            'raw_relation_type_output': responses_relation_type,
            'logic_processed_output': relation_prompt_string_dict
        }, json_file)
    shutil.copy2(prompt_path_entity, out_dir)
    if args.extract_relation_types: 
        shutil.copy2(prompt_path_relation_type, out_dir)

    # get relation rating from llm: {id: relation_response}; relation_response: each line is a relation, followed by the rating
    responses_relation_rating = run_llm_relation(args.api_key, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_relation, test_data, relation_prompt_string_dict)

    # metrics initialization
    counters = [{r.lower(): {'hit': 0, 'num_pred': 0, 'num_true': 0} for r in data['relations']} for _ in range(2)]

    # fail case saving initialization
    correct_cases_list = [{}, {}] # all correct
    fail_cases_list = [{}, {}]
    missing_cases_list = [{}, {}]
    false_cases_list = [{}, {}]
    error_cases_list = [{}, {}]

    for id, sample in tqdm(test_data.items()):
        # get the response of current job; response is a long string, with multiple "relation rating"
        response = responses_relation_rating[id]
        relation_rating_list = response.split('\n')
        sure_response_list = []
        sure_likely_response_list = []
        # classify each relation in current response, by sure/likely, and parse
        for relation_rating in relation_rating_list:
            sure_start = relation_rating.find('yes')
            likely_start = relation_rating.find('likely')
            
            if sure_start >= 1:
                try:
                    string_to_append = tuple(json.loads(relation_rating[:(sure_start-1)].lower()))                    
                    sure_response_list.append(string_to_append)
                    sure_likely_response_list.append(string_to_append)
                except Exception as e:
                    print(f'error in try: {e}')
                    error_cases_list[0][id] = {'error': str(e)}
                    print(f'id: {id}, error response: {response}')
            elif likely_start >= 1:
                try:
                    string_to_append = tuple(json.loads(relation_rating[:(likely_start-1)].lower()))
                    sure_likely_response_list.append(string_to_append)
                except Exception as e:
                    print(f'error in try: {e}')
                    error_cases_list[1][id] = {'error': str(e)}
                    print(f'id: {id}, error response: {response}')

        for list_num, pred_list in enumerate([sure_response_list, sure_likely_response_list]):
            pred_set = set(pred_list)
            true_list = [tuple([item.lower() for item in relation]) for relation in sample['relations']]
            true_set = set(true_list)

            counters[list_num] = update_counter(counters[list_num], true_set, pred_set)
            
            # record fail/correct cases
            record = {'text': sample['text'], 'true': true_list, 'pred': pred_list, 'response': response}
            if pred_set == true_set:
                correct_cases_list[list_num][id] = record
            else:
                fail_cases_list[list_num][id] = record
                if true_set.difference(pred_set):
                    missing_cases_list[list_num][id] = sorted(true_set.difference(pred_set))
                if pred_set.difference(true_set):
                    false_cases_list[list_num][id] = sorted(pred_set.difference(true_set))
            if id in error_cases_list[list_num]:
                error_cases_list[list_num][id].update(record)

    for list_num, scope in enumerate(['sure', 'sure_likely']):
        output = compute_metrics(counters[list_num])
        output.update({
            'num_cases': len(correct_cases_list[list_num]) + len(fail_cases_list[list_num]),
            'accuracy': len(correct_cases_list[list_num])/(len(correct_cases_list[list_num]) + len(fail_cases_list[list_num])),
            'correct_cases': correct_cases_list[list_num], # all correct
            'fail_cases': fail_cases_list[list_num],
            'missing_cases': missing_cases_list[list_num],
            'false_cases': false_cases_list[list_num],
            'error_cases': error_cases_list[list_num]
        })

        with open(os.path.join(out_dir, f'output_{dataset}_seed={args.seed}_split={args.split}_{scope}.json'), 'w') as json_file:
            json.dump(output, json_file)
        shutil.copy2(prompt_path_relation, out_dir)

    # return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--is_async', action='store_true')
    parser.add_argument('--suffix', type=str, default='myFolder')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extract_relation_types', type=bool, default=True)
    parser.add_argument('--check_commonsense', type=bool, default=True)

    parser.add_argument('--dataset', type=str, default='conll04')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--is_val', action='store_true')
    parser.add_argument('--test_k', type=int, default=-1)

    parser.add_argument('--prompt_dir', type=str, default='')

    args = parser.parse_args()

    get_response_from_llm(args)
