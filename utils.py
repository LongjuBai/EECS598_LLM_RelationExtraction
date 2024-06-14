import numpy as np
import time
import asyncio
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, BadRequestError
import openai
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import random
import json
from sklearn.neighbors import NearestNeighbors


def dict_first_k(dic, k):
    return {k: v for k, v in zip(list(dic.keys())[:k], list(dic.values())[:k])}


def dict_random_k(dic, k):
    ids = random.sample(range(len(dic)), k)
    return {k: v for k, v in zip(list(dic.keys())[ids], list(dic.values())[ids])}


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


def run_llm(client, is_async, model, temp, max_tokens, seed, prompt, multi_round, data):
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
        elif model == 'umgpt':
            completion = await client.chat.completions.create(
                model='gpt-35-turbo',
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        else:
            raise Exception('Model Not Supported!')
    
    def llm_worker(id, sample):
        if multi_round:
            messages = dispart_prompt(prompt.replace('$TEXT$', sample['text']))
        else:
            messages = [{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}]
        if model == 'gpt-3.5-turbo-0125':
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
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
        elif model == 'umgpt':
            try:
                completion = client.chat.completions.create(
                    model='gpt-35-turbo',
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    seed=seed
                )
            except BadRequestError as e:
                print(messages)
                return id, ''
            return id, completion.choices[0].message.content
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        responses = dict([llm_worker(id, sample) for id, sample in tqdm(data.items())])
    else:
        responses = dict(asyncio.run(tqdm_asyncio.gather(*[llm_worker_async(id, sample) for id, sample in data.items()]), debug=True))
    return responses


def run_llm_para(client, is_async, model, temp, max_tokens, seed, prompt, data, relation_prompt_string_dict):
    async def llm_worker_async(relation_prompt_string, id, sample):
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
        elif model == 'umgpt':
            completion = await client.chat.completions.create(
                model='gpt-35-turbo',
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": "Use Logic to analyze given text. Be smart."
                    # },
                    # Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary, high school, and colledge knowledge. Answer like a professor, a scholar, and a journalist. 
                    {
                        "role": "user", 
                        "content": prompt.replace('$TEXT$', sample['text'] + relation_prompt_string),
                    }],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        else:
            raise Exception('Model Not Supported!')
    
    def llm_worker(relation_prompt_string, id, sample):
        if model == 'gpt-3.5-turbo-0125':
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": "Use Logic to analyze given text. Be smart. Loc only includes location names shown on map."
                    # },
                    # Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary and high school knowledge.
                    {
                        "role": "user", 
                        "content": prompt.replace('$TEXT$', sample['text']).replace('$ENTITIES$', relation_prompt_string)
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
        elif model == 'umgpt':
            completion = client.chat.completions.create(
                model='gpt-35-turbo',
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": "Use Logic to analyze given text. Be smart."
                    # },
                    # Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary, high school, and colledge knowledge. Answer like a professor, a scholar, and a journalist. 
                    {
                        "role": "user", 
                        "content": prompt.replace('$TEXT$', sample['text'] + relation_prompt_string),
                    }],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        responses = dict([llm_worker(relation_prompt_string_dict[id], id, sample) for id, sample in tqdm(data.items())])
    else:
        responses = dict(asyncio.run(tqdm_asyncio.gather(*[llm_worker_async(relation_prompt_string_dict[id], id, sample) for id, sample in data.items()]), debug=True))
    return responses

# TODO: fix this
def run_llm_relation(client, is_async, model, temp, max_tokens, seed, prompt, data, relation_prompt_string_dict):
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
        elif model == 'umgpt':
            completion = await client.chat.completions.create(
                model='gpt-35-turbo',
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
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
        responses = dict([llm_worker(relation_prompt_string_dict[id], id, sample) for id, sample in tqdm(data.items())])
    else:
        responses = dict(asyncio.run(tqdm_asyncio.gather(*[llm_worker_async(id, sample) for id, sample in data.items()]), debug=True))
    return responses


def run_llm_relation_multi(client, is_async, model, temp, max_tokens, seed, prompt, data, relation_prompt_string_dict):
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
        elif model == 'umgpt':
            completion = await client.chat.completions.create(
                model='gpt-35-turbo',
                messages=[{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
                temperature=temp,
                max_tokens=max_tokens,
                seed=seed
            )
            return id, completion.choices[0].message.content
        else:
            raise Exception('Model Not Supported!')

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(100))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    def llm_worker_multi(relation_prompt_string, id, sample):
        if model == 'gpt-3.5-turbo-0125':
            responses = ''
            for relation in relation_prompt_string:
                completion = completion_with_backoff(
                    model=model,
                    messages=[
                        # {
                        #     "role": "system",
                        #     "content": "Use Logic to analyze given text. Be smart."
                        # },
                        # Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary, high school, and colledge knowledge. Answer like a professor, a scholar, and a journalist. 
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
            responses = ''
            for relation in relation_prompt_string:
                completion = client.completions.create(
                    model=model,
                    prompt=prompt.replace('$TEXT$', sample['text'] + relation),
                    temperature=temp,
                    max_tokens=max_tokens,
                    seed=seed
                )
                responses += completion.choices[0].text
            return id, responses
        elif model == 'umgpt':
            responses = ''
            for relation in relation_prompt_string:
                completion = completion_with_backoff(
                    model='gpt-35-turbo',
                    messages=[
                        # {
                        #     "role": "system",
                        #     "content": "Use Logic to analyze given text. Be smart."
                        # },
                        # Be loyal to the given text content. Use English acitve and passive voice. Use common sense. Use primary, high school, and colledge knowledge. Answer like a professor, a scholar, and a journalist. 
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
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        responses = dict([llm_worker_multi(relation_prompt_string_dict[id], id, sample) for id, sample in tqdm(data.items())])
    else:
        responses = dict(asyncio.run(tqdm_asyncio.gather(*[llm_worker_async(id, sample) for id, sample in data.items()]), debug=True))
    return responses

# TODO: fix this async
# text dict should be: id as key, entity list as value
def run_llm_embed(client, is_async, model, text_dict):
    def llm_worker(id, text_list):
        if model == 'text-embedding-3-large':
            output = {}
            for text in text_list:
                response = client.embeddings.create(
                    input=text,
                    model=model
                )
                output[text] = response.data[0].embedding
            return id, output # output is a dict, with id is the entity, and value is the text embedding. A text embedding is a list itself.
        elif model == 'umgpt':
            output = {}
            for text in text_list:
                response = client.embeddings.create(
                    input=text,
                    model='text-embedding-3-large'
                )
                output[text] = response.data[0].embedding
            return id, output # output is a dict, with id is the entity, and value is the text embedding. A text embedding is a list itself.
        else:
            raise Exception('Model Not Supported!')
    
    if not is_async:
        responses = dict([llm_worker(id, text_list) for id, text_list in tqdm(text_dict.items())])
    else:
        raise Exception("Async is closed for this function")
    return responses # dict, with id as key, and embedding list as value. Note each embedding can be a list itself.


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


def make_client(model, is_async, api_key):
    client_umgpt = (AsyncAzureOpenAI if is_async else AzureOpenAI)(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint = 'https://api.umgpt.umich.edu/azure-openai-api-unlimited',
                organization = '001145'
            )
    client_openai = (AsyncOpenAI if is_async else OpenAI)(api_key=api_key)
    client = client_umgpt if model =='umgpt' else client_openai
    return client


# conll04: 1474 (hate)
def dispart_prompt(prompt):
    messages = []
    prompt = prompt.split('\n\n')
    messages.append({"role": "system", "content": "You are an expert at named entity recognition (NER). " + prompt[0].split(': ')[1]})
    for sample in prompt[1:-2]:
        question, answer = sample.split('\n')
        messages.append({"role": "user", "content": sample.split('\n')[0]})
        messages.append({"role": "assistant", "content": sample.split('\n')[1].split(': ')[-1]})
    messages.append({"role": "user", "content": prompt[-1].split('\n')[0]})
    return messages

# TODO: Handle duplicate context examples
# samples_gt = {id: {'text': ..., 'relations': ..., 'entity/masked sentence': ..., 'embedding': ...}, ...}
def make_icl_prompt(dataset, samples_gt, embeddings, context_len, mode='entity'):
    data = json.load(open(f'datasets598/{dataset}/preprocessed.json', 'r'))['train']

    neigh = NearestNeighbors(n_neighbors=context_len // len(embeddings), n_jobs=-1).fit([sample['embedding'] for sample in samples_gt])
    _, neigh_ids = neigh.kneighbors(embeddings)
    context_samples = [data[id] for id in neigh_ids.flatten()]

    if mode == 'entity':
        prompt = open(f'prompts/{dataset}/prompt_tot_entity.txt', 'r').read()
        prompt = prompt.split('\n\n')
        prompt_new = prompt[0]
        for sample in prompt[1:-2]:
            prompt_new
    elif mode == 'sentence':    
