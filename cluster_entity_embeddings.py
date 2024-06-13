import pickle
import argparse
import os
from sklearn.cluster import DBSCAN, HDBSCAN
import math
from tqdm import tqdm
import json

def cluster_entity_embeddings(args):
    # read in entity embeddings and reformat
    embedding_dir = os.path.join(args.entity_dir, args.suffix, 'entity_embeddings.pickle')
    entity_embeddings_dict = pickle.load(open(embedding_dir, 'rb'))
    entity_embeddings_list = [value for values in entity_embeddings_dict.values() for value in values]

    # auto find optimal the hyperparameters and cluster
    if args.min_cluster_size is not None:
        hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, n_jobs=-1, store_centers="medoid").fit(entity_embeddings_list)
        k = args.min_cluster_size
        print('num of clusters of ', k, 'is', hdb.medoids_.shape[0])
    else:
        entity_count = len(entity_embeddings_list)
        budget = args.relation_count * args.k_shot_per_relation
        min_cluster_size = math.ceil(entity_count / budget)
        for k in tqdm(range(2, min_cluster_size, 1)):
            hdb = HDBSCAN(min_cluster_size=k, n_jobs=-1, store_centers="medoid").fit(entity_embeddings_list)
            print('num of clusters of ', k, 'is', hdb.medoids_.shape[0])
            if hdb.medoids_.shape[0] <= budget:
                break


    # find ids, sentence, entity of medoids as a list
    output_list = []
    id_entity_file_path = os.path.join('outputs', args.dataset+'_train', 'output_conll04_seed=42_split=0_entity.json')
    id_entity_data = json.load(open(id_entity_file_path))
    preprocessed_data_file_path = os.path.join('datasets598', args.dataset, 'preprocessed.json')
    preprocessed_data = json.load(open(preprocessed_data_file_path))
    cluster_centers = [list(embedding) for embedding in hdb.medoids_]
    for id, embedding_list in entity_embeddings_dict.items():
        for embedding_id, embedding in enumerate(embedding_list):
            if list(embedding) in cluster_centers: # list of embeddings (lists)
                obj = { 'id': id, 'sentence': preprocessed_data['train'][id]['text'], 'entity': json.loads(id_entity_data['raw_entity_output'][id])[embedding_id] }
                output_list.append(obj)
                cluster_centers.remove(embedding)
                break; # stop searching current id

    # print and save 
    print(output_list[0])
    print('Totally', args.relation_count, 'relations, with', args.k_shot_per_relation, 'shots per relation => picked min_cluster_size', k)
    print('The clusters found: (cluster_num, embedding_dim): ', hdb.medoids_.shape)
    output_dir = os.path.join(args.entity_dir, args.suffix, args.output_file_name)
    with open(output_dir, 'w') as f:
        # f.write(str(output_list))
        json.dump(output_list, f)
    print('Write as a list of medoids embeddings.')
    return hdb.medoids_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='CONll04_train')

    parser.add_argument('--dataset', type=str, default='conll04')


    parser.add_argument('--k_shot_per_relation', type=int, default=5)

    parser.add_argument('--entity_dir', type=str, default='outputs') # dir to /outputs, no suffix included; suffix specified in --suffix
    parser.add_argument('--output_file_name', type=str, default='entity_embeddings_id_sentence_entity_list.json')
    
    args = parser.parse_args()
    
    args.min_cluster_size = None
    if args.dataset == 'conll04':
        args.relation_count = 5
        args.min_cluster_size = None # 14 # exhaustive search
    elif args.dataset == 'ade':
        args.relation_count = 1
    elif args.dataset == 'nyt':
        args.relation_count = 24 
    cluster_entity_embeddings(args)