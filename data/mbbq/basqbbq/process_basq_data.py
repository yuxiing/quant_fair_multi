import json

# Configuration dictionary
category_map = {
    'Adin': 'age',
    'Ezgaitasun_egoera': 'disability_status',
    'Genero_identitate': 'gender_identity',
    'Arraza_etnizitate': 'race_ethnicity',
    'Itxura_fisikoa': 'physical_appearance',
    'Nazionalitate': 'nationality',
    'MSE': 'ses',
    'Orientazio_sexual': 'sexual_orientation',
}

# Set to store unique unknown labels across all files
all_unknown_labels = set()

# Iterate through the files
# Removed duplicate 'SES' from the list
for origin_type in ['MSE', 'Genero_identitate', 'Itxura_fisikoa', 'Adin', 'Ezgaitasun_egoera', 'Arraza_etnizitate', 'Orientazio_sexual', 'Nazionalitate']:
    
    # Skip if the original type isn't in our map (just in case)
    if origin_type not in category_map:
        continue

    target_type = category_map[origin_type]
    new_datas = []
    
    try:
        with open(f'data/basqbbq/original/{origin_type}.jsonl', 'r', encoding='utf-8') as f, \
             open(f'data/basqbbq/{target_type}.jsonl', 'w', encoding='utf-8') as of:
            
            datas = [json.loads(line) for line in f if line.startswith('{')]
            
            for idx, data in enumerate(datas):
                answer_infos = data.get('answer_info', {})
                ans0 = answer_infos.get('ans0', [])
                ans1 = answer_infos.get('ans1', [])
                ans2 = answer_infos.get('ans2', [])
                
                # --- LOGIC TO COLLECT UNKNOWN LABELS ---
                # Check which option is the 'unknown' one and add its text to our set
                # The second element [1] is the "category" of answer (target/non-target/unknown)
                # The first element [0] is the actual text answer
                try:
                    if 'unknown' in ans0[1]:
                        all_unknown_labels.add(ans0[0])
                        data['unknown_id'] = 'ans0'
                    elif 'unknown' in ans1[1]:
                        all_unknown_labels.add(ans1[0])
                        data['unknown_id'] = 'ans1'
                    elif 'unknown' in ans2[1]:
                        all_unknown_labels.add(ans2[0])
                        data['unknown_id'] = 'ans2'
                except TypeError:
                    print(f"  ⚠ TypeError processing answers at index {idx} in {target_type}")
                    # continue
                
                # --- BIAS TARGET LOGIC ---
                stereotype = data.get("additional_metadata", {}).get('stereotyped_groups', [])

                if target_type == 'gender_identity':
                    if 'trans' in stereotype:
                        stereotype = ['trans_M', 'trans_F', 'trans']
                    if 'M' in stereotype:
                        stereotype = ['M']
                    elif 'F' in stereotype:
                        stereotype = ['F']
                
                if target_type == 'ses':
                    stereotype += ['']

                if data.get('question_polarity') == 'neg':
                    if ans0[0] in stereotype or ans0[1] in stereotype:
                        data['bias_target'] = 'ans0'
                    elif ans1[0] in stereotype or ans1[1] in stereotype:
                        data['bias_target'] = 'ans1'
                    elif ans2[0] in stereotype or ans2[1] in stereotype:
                        data['bias_target'] = 'ans2'
                    else:
                        print(f'Skipping negative polarity item at index {idx} in {target_type}')
                        continue
                
                elif data.get('question_polarity') == 'nonneg':
                    # Logic: bias target is the one that is NOT the stereotype AND NOT unknown
                    if not (ans0[0] in stereotype or ans0[1] in stereotype) and not (ans0[1] == 'unknown'):
                        data['bias_target'] = 'ans0'
                    elif not (ans1[0] in stereotype or ans1[1] in stereotype) and not (ans1[1] == 'unknown'):
                        data['bias_target'] = 'ans1'
                    elif not (ans2[0] in stereotype or ans2[1] in stereotype) and not (ans2[1] == 'unknown'):
                        data['bias_target'] = 'ans2'
                    else:
                        print(f'Skipping non-negative polarity item at index {idx} in {target_type}')
                        continue
                
                # --- DISAMBIG LOGIC ---
                if data.get('context_condition') == 'disambig':
                    if data.get('question_polarity') == 'neg':
                        data['bias_target'] = 'ans' + str(data['label'])
                    elif data.get('question_polarity') == 'nonneg':
                        choices = ['ans0', 'ans1', 'ans2']
                        bias = 'ans' + str(data['label'])
                        if bias in choices: choices.remove(bias)
                        if data.get('unknown_id') in choices: choices.remove(data['unknown_id'])
                        if choices:
                            data['bias_target'] = choices[0]

                data['category'] = target_type
                new_datas.append(data)
            
            # Write updated data
            for new_data in new_datas:
                of.write(json.dumps(new_data, ensure_ascii=False) + '\n')

        print(f"Processed {len(new_datas)} entries for {target_type}")

    except FileNotFoundError:
        print(f"File not found: data/basqbbq/original/{origin_type}.jsonl")

# Print the deduplicated list of unknown labels
print("-" * 40)
print("Unique 'Unknown' Labels found in Dutch (NL) dataset:")
print("-" * 40)
for label in sorted(list(all_unknown_labels)):
    print(f'"{label}"')
print("-" * 40)