import json
import utils
import argparse
import os
import pickle
# entity embedding
# in order to cluster the same entity with variant names
def generate_entity_embedding(args):
    # read in the entity file
    entity_dir = os.path.join(args.entity_dir, args.suffix, args.entity_file_name)
    f = open(entity_dir)
    
    # create client instance
    client = utils.make_client(args.model, args.is_async, args.api_key)

    # with specified dataset, process and send to utils.generate_entity_embed
    if args.dataset == 'conll04':
        entity_dict = json.load(f)['raw_entity_output']
        for key, value in entity_dict.items():
            entity_dict[key] = json.loads(value)
        # print(entity_dict['17'])
        if args.test_k > 0:
            entity_dict = dict([item for item in entity_dict.items()][:args.test_k])
        embedding_dict = utils.run_llm_embed(client, args.is_async, args.model, entity_dict)
        # print(len(embedding_dict['17'][0])) # 3072 embedding
    
    # write the output
    output_dir = os.path.join(args.entity_dir, args.suffix, 'entity_embeddings.pickle')
    # os.makedirs(output_dir, exist_ok=True)
    pickle.dump(embedding_dict, open(output_dir, 'wb'))

    print('- Length of embedding dict', len(embedding_dict))
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='api_key_umgpt')
    parser.add_argument('--is_async', action='store_true')
    parser.add_argument('--suffix', type=str, default='our_final_results')
    parser.add_argument('--entity_file_name', type=str, default='output_conll04_seed=42_split=0_entity.json')

    parser.add_argument('--model', type=str, default='text-embedding-3-large')
    
    parser.add_argument('--dataset', type=str, default='conll04')
    parser.add_argument('--test_k', type=int, default=-1)
    # parser.add_argument('--test_ids', nargs="*", type=int, default=[])

    parser.add_argument('--entity_dir', type=str, default='outputs') # dir to /outputs, no suffix included; suffix specified in --suffix

    args = parser.parse_args()
    args.api_key = open(args.api_key, 'r').read()

    generate_entity_embedding(args)