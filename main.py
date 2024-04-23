from datasets598.utils import load_ade, load_conll04, load_nyt
from utils import dict_first_k, find_triplets, run_llm, update_counter, compute_metrics
import json
import os
from tqdm import tqdm
import argparse


def get_response_from_llm(args):
    dataset = args.dataset

    # load prompt
    if args.is_cot:
        prompt_file_path = os.path.join('prompts', dataset, 'prompt_cot.txt')
    elif args.is_tot:
        prompt_file_path = os.path.join('prompts', dataset, 'prompt_tot.txt')
        prompt_file_path_2 = os.path.join('prompts', dataset, 'prompt_tot2.txt')
    else:
        prompt_file_path = os.path.join('prompts', dataset, 'prompt.txt')
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()
    
    # load test data
    if dataset == 'ade':
        data = load_ade(split=args.split)
        test_data = data['test']
        with open(f'prompts/{dataset}/few_shot_id.txt', 'r') as f:
            for id in f.read().split('\n'):
                test_data.pop(id, None)
    elif dataset == 'conll04':
        data = load_conll04()
        test_data = data['val'] if args.is_val else data['test']
    elif dataset == 'nyt':
        data = load_nyt()
        test_data = data['val'] if args.is_val else data['test']
    else:
        raise Exception('Dataset Not Supported!')
    if args.test_k >= 0:
        test_data = dict_first_k(test_data, args.test_k)

    # get response
    responses = run_llm(args.api_key, args.is_async, args.model, args.temp, args.max_tokens, args.seed, prompt, test_data)

    if args.is_tot:
        responses

    # metrics initialization
    counter = {r.lower(): {'hit': 0, 'num_pred': 0, 'num_true': 0} for r in data['relations']}

    # fail case saving initialization
    correct_cases = {} # all correct
    fail_cases = {}
    missing_cases = {}
    false_cases = {}
    error_cases = {}

    for id, sample in tqdm(test_data.items()):
        response = responses[id]
        pred_list = []
        try:
            pred_list = [tuple(triplet) for triplet in json.loads(find_triplets(response).lower())] # result is a list of triples
        except Exception as e:
            print(f'error in try: {e}')
            error_cases[id] = {'error': str(e)}
            print(f'id: {id}, error response: {response}')
        pred_set = set(pred_list)
        true_list = [tuple([item.lower() for item in relation]) for relation in sample['relations']]
        true_set = set(true_list)

        counter = update_counter(counter, true_set, pred_set)
        
        # record fail/correct cases
        record = {'text': sample['text'], 'true': true_list, 'pred': pred_list, 'response': response}
        if pred_set == true_set:
            correct_cases[id] = record
        else:
            fail_cases[id] = record
            if true_set.difference(pred_set):
                missing_cases[id] = sorted(true_set.difference(pred_set))
            if pred_set.difference(true_set):
                false_cases[id] = sorted(pred_set.difference(true_set))
        if id in error_cases:
            error_cases[id].update(record)

    output = compute_metrics(counter)
    output.update({
        'num_cases': len(correct_cases) + len(fail_cases),
        'accuracy': len(correct_cases)/(len(correct_cases) + len(fail_cases)),
        'correct_cases': correct_cases, # all correct
        'fail_cases': fail_cases,
        'missing_cases': missing_cases,
        'false_cases': false_cases,
        'error_cases': error_cases
    })

    os.makedirs('outputs', exist_ok=True) 
    with open(f'outputs/output_{dataset}_seed={args.seed}_split={args.split}_system_prompt.json', 'w') as json_file:
        json.dump(output, json_file)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='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')
    parser.add_argument('--is_async', action='store_true')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', type=str, default='conll04')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--is_val', action='store_true')
    parser.add_argument('--is_cot', action='store_true')
    parser.add_argument('--test_k', type=int, default=-1)

    args = parser.parse_args()

    get_response_from_llm(args)