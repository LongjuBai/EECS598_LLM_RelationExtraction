# python3 cluster_entity_embeddings.py --suffix CONll04_train
# only run training part
python3 cluster_embeddings.py --dataset conll04 --input_dir outputs/conll04_train --input_file_name entity_embeddings_train_notype.pickle --output_file_name entity_medoids.json