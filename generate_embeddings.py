import json
import utils
import argparse
import os
import pickle


# entity embedding
# in order to cluster the same entity with variant names
def generate_entity_embedding(args):
    # read in the entity file
    # entity_dir = os.path.join(args.entity_dir, args.suffix, args.entity_file_name)
    f = open(args.path_to_entity_file)
    
    # create client instance
    client = utils.make_client(args.model, args.is_async, args.api_key)

    # with specified dataset, process and send to utils.generate_entity_embed
    if args.dataset == 'conll04':
        entity_dict = json.load(f)['raw_entity_output']
        if args.test_k > 0:
            entity_dict = dict([item for item in entity_dict.items()][:args.test_k])
        for key, value in entity_dict.items():
            entity_dict[key] = json.loads(value)
            if args.remove_entity_type:
                entity_dict[key] = [entity.split(':')[0] for entity in entity_dict[key]]
        # print(entity_dict['17'])
        embedding_dict = utils.run_llm_embed(client, args.is_async, args.model, entity_dict)
        # print(len(embedding_dict['17'][0])) # 3072 embedding
    
    # write the output
    # output_dir = os.path.join(args.entity_dir, args.suffix, args.output_file_name)
    # os.makedirs(output_dir, exist_ok=True)
    pickle.dump(embedding_dict, open(args.path_to_output_file, 'wb'))

    print('- Length of embedding dict', len(embedding_dict))
        

def generate_sentence_embedding(args):
    entity_dict = json.load(open(args.path_to_entity_file, 'r'))['raw_entity_output']
    if args.test_k > 0:
        entity_dict = dict([item for item in entity_dict.items()][:args.test_k])
    sentences = json.load(open(args.path_to_sentence_file))[args.part]
    client = utils.make_client(args.model, args.is_async, args.api_key)

    # with specified dataset, process and send to utils.generate_entity_embed
    sentence_dict = {}
    for id, entities in entity_dict.items():
        sentence_dict[id] = sentences[id]['text']
        for entity in json.loads(entities):
            entity, entity_type = entity.split(':')
            sentence_dict[id] = sentence_dict[id].replace(entity, args.entity_type_mapping[entity_type])
        sentence_dict[id] = [sentence_dict[id]]

    embedding_dict = utils.run_llm_embed(client, args.is_async, args.model, sentence_dict)
    
    # write the output
    # output_dir = os.path.join(args.entity_dir, args.suffix, args.output_file_name)
    # os.makedirs(output_dir, exist_ok=True)
    pickle.dump(embedding_dict, open(args.path_to_output_file, 'wb'))

    print('- Length of embedding dict', len(embedding_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='api_key_umgpt')
    parser.add_argument('--is_async', action='store_true')

    # parser.add_argument('--suffix', type=str, default='our_final_results')
    # parser.add_argument('--entity_file_name', type=str, default='output_conll04_seed=42_split=0_entity.json')
    # parser.add_argument('--output_file_name', type=str, default='entity_embeddings.pickle')
    parser.add_argument('--remove_entity_type', action='store_true')

    parser.add_argument('--model', type=str, default='umgpt')
    
    parser.add_argument('--dataset', type=str, default='conll04')
    parser.add_argument('--part', type=str, default='test') # need to find train and test data in the same data file.
    parser.add_argument('--test_k', type=int, default=-1)
    # parser.add_argument('--test_ids', nargs="*", type=int, default=[])
    parser.add_argument('--mode', type=str, default='entity')

    # parser.add_argument('--entity_dir', type=str, default='outputs') # dir to /outputs, no suffix included; suffix specified in --suffix
    parser.add_argument('--path_to_entity_file', type=str, default='outputs/our_final_results/output_conll04_seed=42_split=0_entity.json')
    parser.add_argument('--path_to_sentence_file', type=str, default='datasets598/conll04/preprocessed.json')
    parser.add_argument('--path_to_output_file', type=str, default='outputs/our_final_results/entity_embeddings.pickle')

    args = parser.parse_args()
    args.api_key = open(args.api_key, 'r').read()
    
    if args.mode == 'entity':
        if args.remove_entity_type:
            args.path_to_output_file = args.path_to_output_file.split('.')[0] + '_notype.' + args.path_to_output_file.split('.')[1]
        generate_entity_embedding(args)
    elif args.mode == 'sentence':
        if args.remove_entity_type:
            args.path_to_output_file = args.path_to_output_file.split('.')[0] + '_notype.' + args.path_to_output_file.split('.')[1]
        args.entity_type_mapping = {'Loc': 'Location', 'Per': 'Person', 'Org': 'Organization'}
        generate_sentence_embedding(args)