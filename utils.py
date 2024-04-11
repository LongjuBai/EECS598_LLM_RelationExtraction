import json
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# text = TreebankWordDetokenizer().detokenize(tokens)
import os
from unidecode import unidecode


class Dataset():
    def __init__(self, entities, relations, train, test, val=None):
        self.entities = entities
        self.relations = relations
        self.train = train
        self.test = test
        self.val = val
    
    def remove_few_shots(self, ids):
        for samples in [self.train, self.test, self.val]:
            for i, sample in enumerate(samples):
                if sample['id'] in ids:
                    samples.pop(i)


def load_ade(split=0):
    path = f'datasets/ade/preprocessed_{split}.json'
    if os.path.exists(path):
        return json.load(open(path))
    entities_all = ['Drug', 'Adverse-Effect']
    relations_all = ['Adverse-Effect']
    output = {'entities': entities_all, 'relations': relations_all, 'train': [], 'test': [], 'val': []}
    for part in ['train', 'test']:
        samples = json.load(open(f'datasets/ade/ade_split_{split}_{part}.json'))
        for sample in samples:
            text, entities, relations = sample['tokens'], sample['entities'], sample['relations']
            relations_new = []
            for r in relations:
                ent_1, ent_2 = entities[r['head']], entities[r['tail']]
                relations_new.append([
                    ' '.join(text[ent_1['start']:ent_1['end']]),
                    ' '.join(text[ent_2['start']:ent_2['end']])
                ])
            output[part].append({
                'text': ' '.join(text), 
                'relations': relations_new,
                'id': sample['orig_id']
            })
    json.dump(output, open(path, 'w'))
    return output


def load_conll04():
    path = 'datasets/conll04/preprocessed.json'
    if os.path.exists(path):
        return json.load(open(path))
    entity_dict = {'Loc': 'Loc', 'Org': 'Org', 'Peop': 'Per', 'Other': 'Other'}
    relation_dict = {'Work_For': 'Work For', 'Kill': 'Kill', 'OrgBased_In': 'OrgBased In', 'Live_In': 'Live In', 'Located_In': 'Located In'} 
    output = {'entities': list(entity_dict.values()), 'relations': list(relation_dict.values()), 'train': [], 'test': [], 'val': []}
    for part in ['train', 'test', 'val']:
        samples = json.load(open(f"datasets/conll04/conll04_{part if part != 'val' else 'dev'}.json"))
        for sample in samples:
            text, entities, relations = sample['tokens'], sample['entities'], sample['relations']
            relations_new = []
            for r in relations:
                ent_1, ent_2 = entities[r['head']], entities[r['tail']]
                relations_new.append([
                    f"{' '.join(text[ent_1['start']:ent_1['end']])}:{entity_dict[ent_1['type']]}",
                    relation_dict[r['type']],
                    f"{' '.join(text[ent_2['start']:ent_2['end']])}:{entity_dict[ent_2['type']]}"
                ])
            output[part].append({
                'text': ' '.join(text), 
                'relations': relations_new,
                'id': sample['orig_id']
            })
    json.dump(output, open(path, 'w'))
    return output


def load_nyt():
    path = 'datasets/nyt/preprocessed.json'
    if os.path.exists(path):
        return json.load(open(path))
    entity_dict = {'LOCATION': 'Loc', 'ORGANIZATION': 'Org', 'PERSON': 'Per'}
    relations_all = [r for r in json.load(open('datasets/nyt/relations2id.json')).keys() if r != 'None']
    output = {'entities': list(entity_dict.values()), 'relations': relations_all, 'train': [], 'test': [], 'val': []}
    for part in ['train', 'test', 'val']:
        with open(f"datasets/nyt/raw_{part if part != 'val' else 'valid'}.json") as f:
            for i, line in enumerate(f.readlines()):
                sample = json.loads(line)
                text, entities, relations = sample['sentText'], sample['entityMentions'], sample['relationMentions']
                name2type = {ent['text']: entity_dict[ent['label']] for ent in entities}
                relations_new = []
                for r in relations:
                    r['em1Text'] = unidecode(r['em1Text'])
                    r['em2Text'] = unidecode(r['em2Text'])
                    relations_new.append([
                        f"{r['em1Text']}:{name2type[r['em1Text']]}",
                        r['label'],
                        f"{r['em2Text']}:{name2type[r['em2Text']]}"
                    ])
                output[part].append({
                    'text': text, 
                    'relations': relations_new,
                    'id': i
                })
    json.dump(output, open(path, 'w'))
    return output
