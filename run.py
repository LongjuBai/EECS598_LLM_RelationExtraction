from datasets.utils import load_conll04

def gpt_run_model(args, dataset_file, prompt_file, output_file):
    results = {}
    with open(prompt_file, 'r') as f:
        prompt_ = f.read()
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        
    if 'gpt-4' in args.model_name or "gpt-3.5-turbo-1106" == args.model_name:
        gpt_func = gpt_chat
    else:
        gpt_func = gpt_instruct
    
    source_texts = list(dataset.keys())
    for i in tqdm(range(len(source_texts))):
        # try:
        source_text = source_texts[i]
        prompt = prompt_.replace('$TEXT$', source_text)
        generation = gpt_func(args.model_name, prompt, args.seed)
        # relation_str = post_processing(args.model_name, generation)
        results[source_text] = generation
        if i % 20  == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=6)
        # except:
        #     print(f'error occured at {i}')
        #     continue
        
    return results