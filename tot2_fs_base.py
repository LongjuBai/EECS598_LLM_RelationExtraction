from datasets598.few_shot.utils import load_data, load_info, make_base_prompt
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

    # make prompts
    prompt_relation = make_base_prompt(train_data, info)
    print(prompt_relation)
    
    if args.test_k >= 0:
        test_data = dict_first_k(test_data, args.test_k)
    if args.test_ids:
        test_data = {str(id): test_data[str(id)] for id in args.test_ids}

    # get relation rating from llm: {id: relation_response}; relation_response: each line is a relation, followed by the rating
    print('Run triplet')
    responses_relation = run_llm_relation_base(args.client, args.model, args.temp, args.max_tokens, args.seed, prompt_relation, test_data)

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
        sure_response_list = json.loads(find_triplets_base(responses_relation[id]))

        for list_num, pred_list in enumerate([sure_response_list]):
            pred_list = [tuple([item for item in relation]) for relation in pred_list]
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
            record = {'text': sample['text'], 'true': true_list, 'pred': pred_list}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--api_key', type=str, default='api_key')
    parser.add_argument('--is_async', action='store_true')
    parser.add_argument('--suffix', type=str, default='myFolder')

    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--temp', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='scierc')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--part', default='test')
    parser.add_argument('--test_k', type=int, default=-1)
    parser.add_argument('--test_ids', nargs="*", type=int, default=[])

    args = parser.parse_args()
    args.api_key = open(args.api_key, 'r').read()
    args.client = make_client(args.model, args.is_async, args.api_key)

    get_response_from_llm(args)
