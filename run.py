from utils import load_conll04, load_ade
import json
from tqdm import tqdm
from openai import OpenAI
import os
import numpy as np


def get_response_from_gpt(args = {}, dataset='ade'):
    results = {}
    prompt_file_path = os.path.join('prompts', dataset, 'prompt.txt')
    
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()
    
    test_data = load_ade()['test']

    client = OpenAI(api_key='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')
    
    # metrics initialization
    micro_F1_list = []
    false_pred_num, missing_true_num, total_pred_num, total_true_num = 0, 0, 0, 0

    # fail case saving initialization
    correct_cases = {} # all correct
    fail_cases = {}
    missing_cases = {}
    false_cases = {}
    responses = {}
    for i, (id, data) in tqdm(enumerate(test_data.items())):
        prompt_message = prompt.replace('$TEXT$', data['text']),
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt_message,
            max_tokens=100,
            temperature=0.5
        )
        response = response.choices[0].text
        responses[id] = response

        predicted_list = []
        try:
            predicted_string_processed = response.split('Explanation')[0].replace("'", '"').lower()
            predicted_list = [tuple(relation) for relation in json.loads(predicted_string_processed)] # result is a list of list
        except Exception as e:
            print("error in try: ", e)
            print("LLM message:", response)
        predicted_set = set(predicted_list)
        true_set = set([tuple([item.replace("'", '"').lower() for item in relation]) for relation in data['relations']])

        precision = len(predicted_set.intersection(true_set))/len(predicted_set) if len(predicted_set) > 0 else 0
        recall = len(true_set.intersection(predicted_set))/len(true_set) if len(true_set) > 0 else 0
        micro_F1_list.append(2/(1/precision + 1/recall) if min(precision, recall) > 0 else 0)

        false_pred_num += len(predicted_set.intersection(true_set))
        total_pred_num += len(predicted_set)
        missing_true_num += len(true_set.intersection(predicted_set))
        total_true_num += len(true_set)
        
        # record fail/correct cases
        if predicted_set == true_set:
            correct_cases[id] = list(predicted_set)
        else:
            fail_cases[id] = list(predicted_set)
            missing_cases[id] = list(true_set.difference(predicted_set))
            false_cases[id] = list(predicted_set.difference(true_set))
        
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
        'responses': responses
    }

    with open('output.json', 'w') as json_file:
        json.dump(output, json_file)
        # save if wrong

        # print("text: ", data['text'])
        # print("true relations:", data['relations'])
        # print("result relations", results[id])
    # source_texts = list(dataset.keys())
    # for i in tqdm(range(len(source_texts))):
    #     # try:
    #     source_text = source_texts[i]
    #     prompt = prompt_.replace('$TEXT$', source_text)
    #     generation = gpt_func(args.model_name, prompt, args.seed)
    #     # relation_str = post_processing(args.model_name, generation)
    #     results[source_text] = generation
    #     if i % 20  == 0:
    #         with open(output_file, 'w') as f:
    #             json.dump(results, f, indent=6)
        # except:
        #     print(f'error occured at {i}')
        #     continue
    return results
if __name__ == "__main__":
    get_response_from_gpt()