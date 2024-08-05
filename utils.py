import numpy as np
import time
import asyncio
import pickle
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
    if type(s) is not str:
        return []
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


def run_llm(client, is_async, model, temp, max_tokens, seed, prompt, multi_round, data, dataset, context_length = 10, use_ICL = False):
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
    
    def llm_worker(id, sample, prompt):
        if multi_round:
            messages = dispart_prompt(prompt.replace('$TEXT$', sample['text']))
        else:
            if use_ICL: 
                if dataset == 'conll04':
                    # use entity cluster centers and entity embeddings as shots
                    sample_gt = json.load(open('outputs/conll04_train/entity_medoids.json', 'r'))
                    embeddings = list(pickle.load(open('outputs/conll04_test/entity_embeddings_notype_test.pickle', 'rb'))[id].values())
                    prompt = make_icl_prompt('conll04', sample_gt, embeddings, context_length, mode='entity')
                    messages = [{"role": "user", "content": f"{prompt.replace('$TEXT$', sample['text'])}"}]
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
        responses = dict([llm_worker(id, sample, prompt) for id, sample in tqdm(data.items())])
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
                prompt=prompt.replace('$TEXT$', sample['text']).replace('$ENTITIES$', relation_prompt_string),
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
                        "content": prompt.replace('$TEXT$', sample['text']).replace('$ENTITIES$', relation_prompt_string),
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


def run_llm_relation_multi(client, is_async, model, temp, max_tokens, seed, prompt, data, relation_prompt_string_dict, dataset, use_ICL = False, context_length = 10):
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
                # icl prompts
                if use_ICL:
                    if dataset == 'conll04':
                        sample_gt = json.load(open('outputs/conll04_train/sentence_medoids.json', 'r'))
                        embeddings = list(pickle.load(open('outputs/conll04_test/sentence_embeddings_test.pickle', 'rb'))[id].values())
                        prompt = make_icl_prompt('conll04', sample_gt, embeddings, context_length, mode='sentence')
                    else:
                        pass
                else:
                    pass # use the prompt passed from function interface above
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
                api_version="2024-02-01", #"2024-02-01",
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
    # below: context_len // len(embeddings): desired # of shots // total labeled examples = expected cluster size (each cluster for each shot)
    # note in sentence part, each relation may have negative sampling, so need to divide context_len by 2
    neighbor_num = context_len // len(embeddings) if mode == 'entity' else context_len // (2 * len(embeddings))
    neigh = NearestNeighbors(n_neighbors= max(1, neighbor_num), n_jobs=-1).fit(np.array([sample['embedding'] for sample in samples_gt.values()]))
    _, neigh_ids = neigh.kneighbors(embeddings)
    context_samples = [list(samples_gt.values())[id] for id in neigh_ids.flatten()]
    # print(context_samples)
    # exit(0)
    if mode == 'entity':
        prompt = open(f'prompts/{dataset}/prompt_tot_entity.txt', 'r').read()
        prompt = prompt.split('\n\n')
        prompt_new = prompt[0] + '\n\n' # example instruction
#         prompt_new += '''

# TEXT: "If it does not snow, and a lot, within this month we will have no water to submerge 150, 000 hectares (370, 500 acres) of rice," said Bruno Pusterla, a top official of the Italian Agricultural Confederation.
# Entities: ["Bruno Pusterla:Per", "Italian Agricultural Confederation:Org"]

# TEXT: Meanwhile, Shi Liming at the Institute of Zoology of Kunming found that pandas lack variety in their protein heredity, which may serve as one of the major reasons for pandas' near extinction.
# Entities: ["Shi Liming:Per", "Kunming:Loc", "Institute of Zoology:Org"]

# TEXT: 'The viewers of "JFK" and "The Men Who Killed Kennedy" never learn about these facts, nor do they ever learn about all of the other massive body of evidence that conclusively proves beyond a reasonable doubt that Oswald was the lone gunman who killed President Kennedy and Officer Tippit and that there was no coverup by Earl Warren or by the Warren Commission.;
# Entities: ["Oswald:Per", "President Kennedy:Per", "Officer Tippit:Per", "Earl Warren:Per", "Warren Commission:Org"]

# TEXT: PURCHASE, N.Y.
# Entities: ["PURCHASE:Loc", "N.Y .:Loc"]

# TEXT: BELGRADE, Yugoslavia (AP)
# Entities: ["BELGRADE:Loc", "Yugoslavia:Loc", "AP:Org"]

# TEXT: Rome is in Lazio province and Naples in Campania.
# Entities: ["Rome:Loc", "Lazio:Loc", "Naples:Loc", "Campania:Loc"]

# TEXT: (By ITAR-TASS correspondent Mikhail Shevtsov)
# Entities: ["Mikhail Shevtsov:Per", "ITAR-TASS:Org"]

# TEXT: In the communique, the Group of Rio states that "the Haitian crisis can be resolved only if unrestricted respect is shown for the Governor's Island Agreement which calls for the prompt return of Haitian President Jean Bertrand Aristide to the exercise of his constitutional powers in Haiti.
# Entities: ["Jean Bertrand Aristide:Per", "Haiti:Loc"]

# TEXT: Moscow ITAR-TASS
# Entities: ["Moscow:Loc", "ITAR-TASS:Org"]

# TEXT: King rose to prominence after Mrs. Parks' action in December 1955 in Montgomery, Ala., set the stage for a boycott and subsequent demonstrations that caught the nation by surprise.
# Entities: ["Mrs. Parks:Per", "Montgomery:Loc", "Ala.:Loc"]

# TEXT: Sirhan says he was the lone assassin but can't remember shooting Kennedy.
# Entities: ["Sirhan:Per", "Kennedy:Per"]

# TEXT: In Colorado, 13 inches of snow in Denver Wednesday prompted officials to close Interstate 270 temporarily.
# Entities: ["Colorado:Loc", "Denver:Loc"]

# TEXT: Edward Marks, an official with the Montgomery County Democratic Party, argued that if Ms. Toth is not interested in the job, "she should get out."
# Entities: ["Edward Marks:Per", "Ms. Toth:Per", "Montgomery County:Loc", "Democratic Party:Org"]
# '''
        for i, sample in enumerate(context_samples):
            entities = set()
            for relation in sample['relations']:
                entities.add(relation[0])
                entities.add(relation[-1])
            entities_str = '[' + ', '.join([f'"{entity}"' for entity in entities]) + ']'
            prompt_new += f'TEXT: {sample["text"]}\nEntities: {entities_str}\n\n'
        prompt_new += prompt[-2] + '\n\n' + prompt[-1]
        with open(f'log/prompt_entity_{id}.txt', 'w') as f:
            f.write(prompt_new)
        return prompt_new
    elif mode == 'sentence':    
        # get the question message, with ICL examples for sentences
        # ask ChatGPT
        # get the dataset-specific valid type dict and augment dict first
        if dataset == 'ade':
            valid_type_dict = {
                "Adverse-Effect": set([("Adverse-Effect", "Drug")])
            }
            augment_relation_types = {
                'Adverse-Effect': 'is an adeverse effect for drug'
            }
        elif dataset == 'conll04':
            # get the entity types for each relation type within conll04 dataset
            valid_type_dict = {
                "Work For": set([("Per", "Org")]),
                "Kill": set([("Per", "Per")]),
                "OrgBased In": set([("Org", "Loc")]),
                "Live In": set([("Per", "Loc")]),
                "Located In": set([("Loc", "Loc")])
            }
            augment_relation_types = {
                'Work For': 'Work(ed) For',
                'Kill': 'Kill(ed)',
                'OrgBased In': 'is(was) OrgBased In',
                'Live In': 'Live(d) In',
                'Located In': 'is(was) Located In'
            }
        else:
            raise Exception('Not supported!')

        message = "Example Instructional Prefix: Check if a given pair of entities have relations [OrgBased In, Work For, Located In, Live In, Kill] follows the logic of the given text. Provide a confidence level (Yes/No) for each relation.\n"
#         message += '''

# TEXT: "If it does not snow, and a lot, within this month we will have no water to submerge 150, 000 hectares (370, 500 acres) of rice," said Bruno Pusterla, a top official of the Italian Agricultural Confederation.
# Entities: ["Bruno Pusterla:Per", "Italian Agricultural Confederation:Org"]

# TEXT: Meanwhile, Shi Liming at the Institute of Zoology of Kunming found that pandas lack variety in their protein heredity, which may serve as one of the major reasons for pandas' near extinction.
# Entities: ["Shi Liming:Per", "Kunming:Loc", "Institute of Zoology:Org"]

# TEXT: 'The viewers of "JFK" and "The Men Who Killed Kennedy" never learn about these facts, nor do they ever learn about all of the other massive body of evidence that conclusively proves beyond a reasonable doubt that Oswald was the lone gunman who killed President Kennedy and Officer Tippit and that there was no coverup by Earl Warren or by the Warren Commission.;
# Entities: ["Oswald:Per", "President Kennedy:Per", "Officer Tippit:Per", "Earl Warren:Per", "Warren Commission:Org"]

# TEXT: PURCHASE, N.Y.
# Entities: ["PURCHASE:Loc", "N.Y .:Loc"]

# TEXT: BELGRADE, Yugoslavia (AP)
# Entities: ["BELGRADE:Loc", "Yugoslavia:Loc", "AP:Org"]

# TEXT: Rome is in Lazio province and Naples in Campania.
# Entities: ["Rome:Loc", "Lazio:Loc", "Naples:Loc", "Campania:Loc"]

# TEXT: (By ITAR-TASS correspondent Mikhail Shevtsov)
# Entities: ["Mikhail Shevtsov:Per", "ITAR-TASS:Org"]

# TEXT: In the communique, the Group of Rio states that "the Haitian crisis can be resolved only if unrestricted respect is shown for the Governor's Island Agreement which calls for the prompt return of Haitian President Jean Bertrand Aristide to the exercise of his constitutional powers in Haiti.
# Entities: ["Jean Bertrand Aristide:Per", "Haiti:Loc"]

# TEXT: Moscow ITAR-TASS
# Entities: ["Moscow:Loc", "ITAR-TASS:Org"]

# TEXT: King rose to prominence after Mrs. Parks' action in December 1955 in Montgomery, Ala., set the stage for a boycott and subsequent demonstrations that caught the nation by surprise.
# Entities: ["Mrs. Parks:Per", "Montgomery:Loc", "Ala.:Loc"]

# TEXT: Sirhan says he was the lone assassin but can't remember shooting Kennedy.
# Entities: ["Sirhan:Per", "Kennedy:Per"]

# TEXT: In Colorado, 13 inches of snow in Denver Wednesday prompted officials to close Interstate 270 temporarily.
# Entities: ["Colorado:Loc", "Denver:Loc"]

# TEXT: Edward Marks, an official with the Montgomery County Democratic Party, argued that if Ms. Toth is not interested in the job, "she should get out."
# Entities: ["Edward Marks:Per", "Ms. Toth:Per", "Montgomery County:Loc", "Democratic Party:Org"]
# '''
        # get the ground truth in the training examples
        # set the true relations as Yes answers
        # if reverse gives the wrong relations, set as No answers
        for example in context_samples: # example is {'text': ..., 'relations': ..., 'entity/masked sentence': ..., 'embedding': ...}
            relations = example['relations'] # a list of relations
            for relation in relations:
                if dataset == 'conll04':
                    subject = relation[0][:-4]
                    subject_type = relation[0][-3:]
                    verb = relation[1]
                    object = relation[2][:-4]
                    object_type = relation[0][-3:]
                    message += '\n'
                    message += "Given the text: " + example['text'] + '\n'
                    message += subject + ' ' + augment_relation_types[verb] + ' ' + object + ', Yes or No?\n'
                    message += "Answer:\n"
                    message += "Yes\n"

                    # negative sampling
                    if (relation[2][-3:], relation[0][-3:]) in valid_type_dict[relation[1]]:
                        message += '\n'
                        message += "Given the text: " + example['text'] + '\n'
                        message +=  object + ' ' + augment_relation_types[relation[1]] + ' ' + subject + ', Yes or No?\n'
                        message += "Answer:\n"
                        message += "No\n"
                elif dataset == 'ade':
                    subject = relation[0]
                    # subject_type = "Drug"
                    verb = "Adverse-Effect"
                    object = relation[1]
                    # object_type = "Adverse-Effect"
                    message += '\n'
                    message += subject + verb + object + ', Yes or No?\n'
                else:
                    raise Exception("Not supported")

        if dataset == 'conll04':
            message += '\n'
            message += 'Instructional Prefix: Check if a given pair of entities have relations [OrgBased In, Work For, Located In, Live In, Kill] follows the logic of the given text. Provide a confidence level (yes/no) for each relation.\n'
            message += '\n'
        elif dataset == 'ade':
            pass
        message += 'Given the text: $TEXT$\n'
        message += 'Answer:'
        with open(f'log/prompt_sentence_{id}.txt', 'w') as f:
            f.write(message)
        return message





        