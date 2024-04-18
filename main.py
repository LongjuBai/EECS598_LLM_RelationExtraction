from utils import load_ade, load_conll04, load_nyt
import json
from tqdm import tqdm
from openai import OpenAI
import os
import numpy as np
import argparse


def get_response_from_gpt(args):
    dataset, split, seed, api_key = args.dataset, args.split, args.seed, args.key

    prompt_file_path = os.path.join('prompts', dataset, 'prompt.txt')
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()
    
    if dataset == 'ade':
        test_data = load_ade(split=split)['test']
        with open(f'prompts/{dataset}/few_shot_id.txt', 'r') as f:
            for id in f.read().split('\n'):
                test_data.pop(id, None)
    elif dataset == 'conll04':
        test_data = load_conll04()['test']
    elif dataset == 'nyt':
        test_data = load_nyt()['test']
    else:
        raise Exception('Dataset Not Supported!')

    client = OpenAI(api_key=api_key)
    
    # metrics initialization
    micro_F1_list = []
    false_pred_num, missing_true_num, total_pred_num, total_true_num = 0, 0, 0, 0

    # fail case saving initialization
    correct_cases = {} # all correct
    fail_cases = {}
    missing_cases = {}
    false_cases = {}
    for i, (id, data) in tqdm(enumerate(test_data.items())):
        prompt_message = prompt.replace('$TEXT$', data['text']),
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt_message,
            max_tokens=1000,
            temperature=0.5,
            seed=seed
        )
        response = response.choices[0].text

        pred_list = []
        try:
            pred_string_processed = response.split('Explanation')[0].replace("'", '"').lower()
            pred_list = [tuple(relation) for relation in json.loads(pred_string_processed)] # result is a list of list
        except Exception as e:
            print('error in try: ', e)
            print(f'id: {id}, LLM message: {response}')
        pred_set = set(pred_list)
        true_list = [tuple([item.replace("'", '"').lower() for item in relation]) for relation in data['relations']]
        true_set = set(true_list)

        precision = len(pred_set.intersection(true_set))/len(pred_set) if len(pred_set) > 0 else 0
        recall = len(true_set.intersection(pred_set))/len(true_set) if len(true_set) > 0 else 0
        micro_F1_list.append(2/(1/precision + 1/recall) if min(precision, recall) > 0 else 0)

        false_pred_num += len(pred_set.intersection(true_set))
        total_pred_num += len(pred_set)
        missing_true_num += len(true_set.intersection(pred_set))
        total_true_num += len(true_set)
        
        # record fail/correct cases
        record = {'text': data['text'], 'true': true_list, 'pred': pred_list, 'response': response}
        if pred_set == true_set:
            correct_cases[id] = record
        else:
            fail_cases[id] = record
            if true_set.difference(pred_set):
                missing_cases[id] = sorted(true_set.difference(pred_set))
            if pred_set.difference(true_set):
                false_cases[id] = sorted(pred_set.difference(true_set))
        
        # early termination
        if i >= 5:
            break

    precision = false_pred_num/total_pred_num
    recall = missing_true_num/total_true_num
    macro_F1 = 2/(1/precision + 1/recall)
    micro_F1 = np.mean(micro_F1_list)

    output = {
        'precision': precision,
        'recall': recall,
        'macro_F1': macro_F1,
        'micro_F1': micro_F1,
        'accuracy': len(correct_cases)/(len(correct_cases) + len(fail_cases)),
        'correct_cases': correct_cases, # all correct
        'fail_cases': fail_cases,
        'missing_cases': missing_cases,
        'false_cases': false_cases,
    }

    os.makedirs('outputs', exist_ok=True) 
    with open(f'outputs/output_{dataset}_seed={seed}_split={split}.json', 'w') as json_file:
        json.dump(output, json_file)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ade')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--key', type=str, default='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')
    args = parser.parse_args()

    get_response_from_gpt(args)