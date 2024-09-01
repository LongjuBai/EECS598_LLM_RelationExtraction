from datasets598.few_shot.utils import load_data, load_info, make_entity_prompt, make_relation_prompt, make_triplet_prompt
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

    # load test data
    data = load_data(dataset, split=args.split, n_shot=args.n_shot)

    train_data, test_data = data['train'], data['test']

    info = load_info(dataset)
    valid_type_dict = info['entity relation map']
    augment_relation_types = info['augmented relation types']

    # make prompts
    prompt_entity = make_entity_prompt(train_data, info)
    print(prompt_entity)
    if args.relation_type_extraction:
        prompt_relation_type = make_relation_prompt(train_data, info)
        print(prompt_relation_type)
    prompt_relation = make_triplet_prompt(train_data, info, args.single_choice, args.compact)
    print(prompt_relation)
    
    if args.test_k >= 0:
        test_data = dict_first_k(test_data, args.test_k)
    if args.test_ids:
        test_data = {str(id): test_data[str(id)] for id in args.test_ids}

    # get response; {id: response}
    print('Run entity')
    responses_entity = run_llm(args.client, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_entity, args.multi_round, test_data, dataset=args.dataset, use_ICL=False, context_length=args.number_of_shots_for_entity_extraction)
    if args.relation_type_extraction:
        print('Run relation') 
        responses_relation_type = run_llm(args.client, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_relation_type, args.multi_round, test_data, dataset=args.dataset, use_ICL=False, context_length=args.number_of_shots_for_entity_extraction)

    # logic to map the extracted entities / relation types into relations, with the valid type dict
    relation_prompt_string_dict = {} # {id: prompt string for candidate relations}
    relation_answer_string_dict = {}
    for id, response in responses_entity.items():

        # try to parse each response; if fail, pass an empty list
        entity_type_dict = struct_response_entity(response, info['entity types'])
        relation_type_list = struct_response_relation(responses_relation_type[id]) if args.relation_type_extraction else info['relation types']

        # construct relations; {relation type: [[ins1, int2], [ins3, ins4]]}
        relation_type_dict = {}
        for relation_type in relation_type_list:
            if relation_type not in valid_type_dict:
                continue
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
        relation_prompt_string = []
        relation_answer_string = []
        all_entity_pairs = set()
        for relation_type, entity_pairs in relation_type_dict.items():
            for entity_pair in entity_pairs:
                entity_1, entity_2 = entity_pair[0].split(':')[0][1:], entity_pair[1].split(':')[0][1:]
                entity_1_type, entity_2_type = entity_pair[0].split(':')[1][:-1], entity_pair[1].split(':')[1][:-1]
                if args.single_choice:
                    relation_prompt_string.append(f'\n{entity_1} {augment_relation_types[relation_type]} {entity_2}, Yes or No?')
                    relation_answer_string.append((entity_pair[0].strip('"'), f'{relation_type}', entity_pair[1].strip('"')))
                else:
                    if tuple(entity_pair) not in all_entity_pairs:
                        all_entity_pairs.add(tuple(entity_pair))
                        prompt_s = '\n'
                        answer_s = {}
                        for relation_type in info['relation entity map'][(entity_1_type, entity_2_type)]:
                            prompt_s += f'\n("{relation_type}") {entity_1} {augment_relation_types[relation_type]} {entity_2}'
                            answer_s[relation_type] = ((entity_pair[0].strip('"'), f'{relation_type}', entity_pair[1].strip('"')))
                        prompt_s += f'\n("none") None of the above'
                        relation_prompt_string.append(prompt_s)
                        relation_answer_string.append(answer_s)

        # record to the dictionary
        relation_prompt_string_dict[id] = relation_prompt_string
        relation_answer_string_dict[id] = relation_answer_string
    
    # save the entity output
    with open(os.path.join(out_dir, f'output_{dataset}_seed={args.seed}_split={args.split}_entity.json'), 'w') as json_file:
        json.dump({
            'raw_entity_output': responses_entity,
            # 'raw_relation_type_output': responses_relation_type,
            'logic_processed_output': relation_prompt_string_dict
        }, json_file)
    # if out_dir != args.prompt_dir:
    #     shutil.copy2(prompt_path_entity, out_dir)
    # if args.relation_type_extraction and out_dir != args.prompt_dir: 
    #     shutil.copy2(prompt_path_relation_type, out_dir)
    
    if args.do_paraphrase:
        responses_para = run_llm_para(args.client, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_para, test_data, responses_entity)
        for id in test_data:
            test_data[id]['text'] = responses_para[id]
            # print(test_data[id]['text'])
        # if out_dir != args.prompt_dir:
        #     shutil.copy2(prompt_path_para, out_dir)

    # get relation rating from llm: {id: relation_response}; relation_response: each line is a relation, followed by the rating
    print('Run triplet')
    responses_relation_rating = run_llm_relation_multi(args.client, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt_relation, test_data, relation_prompt_string_dict, dataset=args.dataset, context_length=args.number_of_shots_for_relation, compact=args.compact)

    # metrics initialization
    counters = [{r: {'hit': 0, 'num_pred': 0, 'num_true': 0} for r in info['relation types']} for _ in range(2)]

    # fail case saving initialization
    correct_cases_list = [{}, {}] # all correct
    fail_cases_list = [{}, {}]
    missing_cases_list = [{}, {}]
    false_cases_list = [{}, {}]
    error_cases_list = [{}, {}]

    for id, sample in tqdm(test_data.items()):
        # get the response of current job; response is a long string, with multiple "relation rating"
        response = responses_relation_rating[id]
        relation_rating_list = response.strip().split('\n')
        sure_response_list = []
        # classify each relation in current response, by sure/likely, and parse
        print(relation_rating_list)
        for relation_i, relation_rating in enumerate(relation_rating_list):
            if args.single_choice:
                sure_start = relation_rating.find('Yes')
                if sure_start >= 0:
                    try:
                        string_to_append = relation_answer_string_dict[id][relation_i]
                        sure_response_list.append(string_to_append)
                    except Exception as e:
                        # print(f'error in try: {e}')
                        error_cases_list[0][id] = {'error': str(e)}
                        # print(f'id: {id}, error response: {response}')
            else:
                for r in info['relation types']:
                    sure_start = relation_rating.find(r)
                    if sure_start >= 0:
                        try:
                            string_to_append = relation_answer_string_dict[id][relation_i][r]
                            sure_response_list.append(string_to_append)
                        except Exception as e:
                            # print(f'error in try: {e}')
                            error_cases_list[0][id] = {'error': str(e)}
                            # print(f'id: {id}, error response: {response}')

        for list_num, pred_list in enumerate([sure_response_list]):
            pred_set = set(pred_list)
            true_list = [tuple([item for item in relation]) for relation in sample['relations']]
            true_set = set(true_list)

            # handel undirected relation
            if dataset == 'scierc':
                for s0 in [pred_set, true_set]:
                    s = s0.copy()
                    s_part = {'compare': set(), 'conjunction': set()}
                    for r in s:
                        if r[1] in s_part:
                            if tuple(sorted((r[0], r[2]))) in s_part[r[1]]:
                                s0.remove(r)
                            else:
                                s_part[r[1]].add(tuple(sorted((r[0], r[2]))))

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

    for list_num, scope in enumerate(['sure']):
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
        # if out_dir != args.prompt_dir:
        #     shutil.copy2(prompt_path_relation, out_dir)

    # return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--api_key', type=str, default='api_key')
    parser.add_argument('--is_async', action='store_true')
    parser.add_argument('--suffix', type=str, default='myFolder')

    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--temp', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--relation_type_extraction', action='store_true')
    parser.add_argument('--do_paraphrase', action='store_true')
    parser.add_argument('--check_commonsense', action='store_true')
    parser.add_argument('--multi_round', action='store_true')
    parser.add_argument('--compact', action='store_true')

    parser.add_argument('--dataset', type=str, default='scierc')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--part', default='test')
    parser.add_argument('--test_k', type=int, default=-1)
    parser.add_argument('--test_ids', nargs="*", type=int, default=[])
    parser.add_argument('--number_of_shots_for_entity_extraction', type=int, default=10)
    # parser.add_argument('--number_of_shots_for_paraphrase', type=int, default=3)
    parser.add_argument('--number_of_shots_for_relation', type=int, default=10)

    parser.add_argument('--prompt_dir', type=str, default='')

    args = parser.parse_args()
    args.api_key = open(args.api_key + '_umgpt', 'r').read() if args.model == 'umgpt' else open(args.api_key, 'r').read()
    args.client = make_client(args.model, args.is_async, args.api_key)
    args.single_choice = args.dataset == 'conll04'

    get_response_from_llm(args)
