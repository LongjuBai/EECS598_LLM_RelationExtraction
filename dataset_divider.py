import os
import json
import shutil


# Define the main folder and relations
relations = ["used for", "feature of", "compare", "evaluated for", "conjunction", "hyponym of", "part of"]
main_folder = 'scierc'
output_folders = [f'scierc{i}' for i in range(7)]

# Create 7 copies of the main folder for each relation
for i, relation in enumerate(relations):
    shutil.copytree(main_folder, output_folders[i])

def process_preprocessed_json(file_path, target_relation):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for split in ["train", "val", "test"]:
        for sample_key in list(data[split].keys()):
            # Filter the relations
            data[split][sample_key]["relations"] = [
                rel for rel in data[split][sample_key]["relations"] if rel[1] == target_relation
            ]
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def process_relation_schema(file_path, target_relation):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Modify the lines to only keep the target_relation
    # Update the list of relations
    lines[0] = f'["{target_relation}"]\n'
    # Keep the rest (types)
    # Update the relation-specific dictionary
    lines[2] = f'{{"{target_relation}": []}}\n'
    
    with open(file_path, 'w') as f:
        f.writelines(lines)

def process_record_schema(file_path, target_relation):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # The first list (entity types) should remain unchanged
    entity_types = lines[0]  # Entity types (e.g., ["generic", "method", ...])
    # Modify the second list (relations) to keep only the target relation
    lines[1] = f'["{target_relation}"]\n'

    # Modify the dictionary to keep only the target relation for each entity type
    entity_types_list = json.loads(entity_types)  # Convert the string to a list

    # Now construct the new dictionary part, preserving all entity types but keeping only the target relation
    new_mappings = {entity_type: [target_relation] for entity_type in entity_types_list}
    
    # Convert the dictionary back to a JSON string but without pretty-printing (compact format)
    new_mappings_str = json.dumps(new_mappings)

    # Write the modified lines back to the file
    with open(file_path, 'w') as f:
        f.writelines([entity_types, lines[1], new_mappings_str + '\n'])


def modify_files_for_relation(relation_index, relation_name):
    for seed_folder in [f'seed{i}' for i in range(1, 11)]:
        for shot_folder in ['1shot', '5shot', '10shot']:
            base_path = os.path.join(output_folders[relation_index], seed_folder, shot_folder)
            
            # File paths for the three files
            preprocessed_json_path = os.path.join(base_path, 'preprocessed.json')
            relation_schema_path = os.path.join(base_path, 'relation.schema')
            record_schema_path = os.path.join(base_path, 'record.schema')
            
            # Process each file
            process_preprocessed_json(preprocessed_json_path, relation_name)
            process_relation_schema(relation_schema_path, relation_name)
            process_record_schema(record_schema_path, relation_name)


# Loop through each relation and modify the corresponding dataset
for i, relation in enumerate(relations):
    modify_files_for_relation(i, relation)

print("Processing complete!")
