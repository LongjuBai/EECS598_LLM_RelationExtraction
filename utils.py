import numpy as np
import asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio


def dict_first_k(dic, k):
    return {k: v for k, v in zip(list(dic.keys())[:k], list(dic.values())[:k])}


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
                messages=[{"role": "system", "content": "You are a linguist. You are good at parsing sentences correctly."}, 
                          {"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}],
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


def compute_metrics(counter):
    def compute_f1(pc, rc):
        if pc == 0 or rc == 0:
            return 0
        return 2 / (1 / pc + 1 / rc)

    pc_list = {r: counter[r]['hit'] / counter[r]['num_pred'] if counter[r]['num_pred'] else 0 for r in counter}
    micro_pc = np.mean(list(pc_list.values()))
    macro_pc = sum([counter[r]['hit'] for r in counter]) / sum([counter[r]['num_pred'] for r in counter])
    
    rc_list = {r: counter[r]['hit'] / counter[r]['num_true'] if counter[r]['num_true'] else 0 for r in counter}
    micro_rc = np.mean(list(rc_list.values()))
    macro_rc = sum([counter[r]['hit'] for r in counter]) / sum([counter[r]['num_true'] for r in counter])
    
    f1_list = {r: compute_f1(pc_list[r], rc_list[r]) for r in counter}
    micro_f1 = np.mean([compute_f1(pc_list[r], rc_list[r]) for r in counter])
    macro_f1 = compute_f1(macro_pc, macro_rc)

    return {'micro_pc': micro_pc, 'macro_pc': macro_pc, 'pc_list': pc_list,
            'micro_rc': micro_rc, 'macro_rc': macro_rc, 'rc_list': rc_list,
            'micro_f1': micro_f1, 'macro_f1': macro_f1, 'f1_list': f1_list}