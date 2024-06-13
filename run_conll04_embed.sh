# conll04 test
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_test
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_test --remove_entity_type
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_test --mode sentence --part test


# conllo4 train
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_train
python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_train --remove_entity_type
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_train --mode sentence --part train