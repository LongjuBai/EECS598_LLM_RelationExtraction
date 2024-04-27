import numpy as np
import asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio


def dict_first_k(dic, k):
    return {k: v for k, v in zip(list(dic.keys())[:k], list(dic.values())[:k])}


def parse_list_string(s, marks_removed=''):
    start, end = s.find('['), s.find(']')
    if start == -1 or end == -1:
        return []
    item_list = s[start+1:end].split(',')
    return [item.strip().strip(marks_removed) for item in item_list]


def struct_response_entity(response, entity_types):
    def find_entity_type(s):
        return s.split(':')[-1][:-1]
    entity_type_dict = {entity_type: [] for entity_type in entity_types}
    entity_list = parse_list_string(response) # ['"fds:Per"', '"ior:Loc"', '"vio:Org"']
    for entity in entity_list:
        entity_type = find_entity_type(entity)
        if entity_type in entity_type_dict:
            entity_type_dict[entity_type].append(entity)
    return entity_type_dict


def struct_response_relation(response):
    relation_list = parse_list_string(response, marks_removed='"') # ['Live In', 'Kill', 'Located In']
    return relation_list


def find_triplets(s):
    start, end = s.find('[['), s.find(']]')
    if start == -1 or end == -1:
        return ''
    return s[start:end+2]


def run_llm(api_key, is_async, model, temp, max_tokens, seed, prompt, data):
    async def llm_worker_async(id, sample):
        if model == 'gpt-3.5-turbo-0125':
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        elif model == 'gpt-3.5-turbo-instruct' or model == 'davinci-002':
            completion = await client.completions.create(
                model=model,
                prompt=prompt.replace('$TEXT$', sample['text']),
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].text
        else:
            raise Exception('Model Not Supported!')
    
    def llm_worker(id, sample):
        if model == 'gpt-3.5-turbo-0125':
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        elif model == 'gpt-3.5-turbo-instruct' or model == 'davinci-002':
            completion = client.completions.create(
                model=model,
                prompt=prompt.replace('$TEXT$', sample['text']),
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].text
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        client = OpenAI(api_key=api_key)
        responses = dict([llm_worker(id, sample) for id, sample in tqdm(data.items())])
    else:
        client = AsyncOpenAI(api_key=api_key)
        loop = asyncio.get_event_loop()
        responses = dict(loop.run_until_complete(tqdm_asyncio.gather(*[llm_worker_async(id, sample) for id, sample in data.items()])))
        loop.close()
    return responses

def run_llm_relation(api_key, is_async, model, temp, max_tokens, seed, prompt, data, relation_prompt_string_dict):
    async def llm_worker_async(id, sample):
        if model == 'gpt-3.5-turbo-0125':
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        elif model == 'gpt-3.5-turbo-instruct' or model == 'davinci-002':
            completion = await client.completions.create(
                model=model,
                prompt=prompt.replace('$TEXT$', sample['text']),
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].text
        else:
            raise Exception('Model Not Supported!')
    
    def llm_worker(relation_prompt_string, id, sample):
        if model == 'gpt-3.5-turbo-0125':
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Use Logic to analyze given text. Be smart. Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary and high school knowledge."
                    },
                    {
                        "role": "user", 
                        "content": prompt.replace('$TEXT$', sample['text'] + relation_prompt_string)
                    }],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        elif model == 'gpt-3.5-turbo-instruct' or model == 'davinci-002':
            completion = client.completions.create(
                model=model,
                prompt=prompt.replace('$TEXT$', sample['text'] + relation_prompt_string),
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].text
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        client = OpenAI(api_key=api_key)
        responses = dict([llm_worker(relation_prompt_string_dict[id], id, sample) for id, sample in tqdm(data.items())])
    else:
        client = AsyncOpenAI(api_key=api_key)
        loop = asyncio.get_event_loop()
        responses = dict(loop.run_until_complete(tqdm_asyncio.gather(*[llm_worker_async(id, sample) for id, sample in data.items()])))
        loop.close()
    return responses


def run_llm_relation_multi(api_key, is_async, model, temp, max_tokens, seed, prompt, data, relation_prompt_string_dict):
    async def llm_worker_async(id, sample):
        if model == 'gpt-3.5-turbo-0125':
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        elif model == 'gpt-3.5-turbo-instruct' or model == 'davinci-002':
            completion = await client.completions.create(
                model=model,
                prompt=prompt.replace('$TEXT$', sample['text']),
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].text
        else:
            raise Exception('Model Not Supported!')

    def llm_worker_multi(relation_prompt_string, id, sample):
        if model == 'gpt-3.5-turbo-0125':
            responses = ''
            for relation in relation_prompt_string:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Use Logic to analyze given text. Be smart. Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary, high school, and colledge knowledge. Answer like a professor, a scholar, and a journalist. "
                        },
                        {
                            "role": "user", 
                            "content": prompt.replace('$TEXT$', sample['text'] + relation)
                        }],
                    temperature=temp,
                    max_tokens=max_tokens,
                    seed=seed
                )
                responses += completion.choices[0].message.content + '\n'
            return id, responses
        elif model == 'gpt-3.5-turbo-instruct' or model == 'davinci-002':
            completion = client.completions.create(
                model=model,
                prompt=prompt.replace('$TEXT$', sample['text'] + relation_prompt_string),
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].text
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        client = OpenAI(api_key=api_key)
        responses = dict([llm_worker_multi(relation_prompt_string_dict[id], id, sample) for id, sample in tqdm(data.items())])
    else:
        client = AsyncOpenAI(api_key=api_key)
        loop = asyncio.get_event_loop()
        responses = dict(loop.run_until_complete(tqdm_asyncio.gather(*[llm_worker_async(id, sample) for id, sample in data.items()])))
        loop.close()
    return responses


def update_counter(counter, true_set, pred_set):
    if len(counter) > 1:
        for triplet in pred_set:
            if triplet[1] in counter:
                counter[triplet[1]]['num_pred'] += 1
        for triplet in true_set:
            if triplet[1] in counter:
                counter[triplet[1]]['num_true'] += 1
        for triplet in pred_set.intersection(true_set):
            if triplet[1] in counter:
                counter[triplet[1]]['hit'] += 1
    else:
        r = list(counter.keys())[0]
        counter[r]['num_pred'] += len(pred_set)
        counter[r]['num_true'] += len(true_set)
        counter[r]['hit'] += len(pred_set.intersection(true_set))
    return counter


def compute_metrics(counter):
    def compute_f1(pc, rc):
        if pc == 0 or rc == 0:
            return 0
        return 2 / (1 / pc + 1 / rc)

    pc_list = {r: counter[r]['hit'] / counter[r]['num_pred'] if counter[r]['num_pred'] else 0 for r in counter}
    micro_pc = np.mean(list(pc_list.values()))
    macro_pc = sum([counter[r]['hit'] for r in counter]) / temp if (temp := sum([counter[r]['num_pred'] for r in counter])) != 0 else 0
    
    rc_list = {r: counter[r]['hit'] / counter[r]['num_true'] if counter[r]['num_true'] else 0 for r in counter}
    micro_rc = np.mean(list(rc_list.values()))
    macro_rc = sum([counter[r]['hit'] for r in counter]) / temp if (temp := sum([counter[r]['num_true'] for r in counter])) != 0 else 0
    
    f1_list = {r: compute_f1(pc_list[r], rc_list[r]) for r in counter}
    micro_f1 = np.mean([compute_f1(pc_list[r], rc_list[r]) for r in counter])
    macro_f1 = compute_f1(macro_pc, macro_rc)

    return {'micro_pc': micro_pc, 'macro_pc': macro_pc, 'pc_list': pc_list,
            'micro_rc': micro_rc, 'macro_rc': macro_rc, 'rc_list': rc_list,
            'micro_f1': micro_f1, 'macro_f1': macro_f1, 'f1_list': f1_list}