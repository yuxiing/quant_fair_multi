import json

# Configuration dictionary
category_map = {
    'Age': 'age',
    'Disability_status': 'disability_status',
    'Gender_identity': 'gender_identity',
    'Language_Formality': 'language_formality',
    'Religion': 'religion',
    'Physical_appearance': 'physical_appearance',
    'Regional':'regional',
    'SES': 'ses',
}

# Set to store unique unknown words
unknown_words_set = set()

# Iterate through the files
for origin_type in ['SES', 'Gender_identity', 'Physical_appearance', 'Age', 'Disability_status', 'Regional','Language_Formality', 'Religion']:
    
    # Skip if the original type isn't in our map
    if origin_type not in category_map:
        continue

    target_type = category_map[origin_type]
    new_datas = []
    
    try:
        with open(f'data/pakbbq/original/{origin_type}.jsonl', 'r', encoding='utf-8') as f, \
             open(f'data/pakbbq/{target_type}.jsonl', 'w', encoding='utf-8') as of:
            
            datas = [json.loads(line) for line in f if line.strip()]
            
            for idx, data in enumerate(datas):
                answer_infos = data.get('answer_info', {})
                ans0 = answer_infos.get('ans0', [])
                ans1 = answer_infos.get('ans1', [])
                ans2 = answer_infos.get('ans2', [])
                
                # --- LOGIC TO COLLECT UNKNOWN WORDS ---
                # Checks which answer option is 'unknown' and adds its Urdu text to the set
                try:
                    if len(ans0) > 1 and 'unknown' in ans0[1]:
                        unknown_words_set.add(data['ans0'])
                        data['unknown_id'] = 'ans0'
                    elif len(ans1) > 1 and 'unknown' in ans1[1]:
                        unknown_words_set.add(data['ans1'])
                        data['unknown_id'] = 'ans1'
                    elif len(ans2) > 1 and 'unknown' in ans2[1]:
                        unknown_words_set.add(data['ans2'])
                        data['unknown_id'] = 'ans2'
                except (TypeError, IndexError):
                    print(f"  ⚠ Error processing answers at index {idx} in {target_type}")
                    continue
                
                # --- BIAS TARGET LOGIC (User's original logic) ---
                stereotype = data.get("additional_metadata", {}).get('stereotyped_groups', [])

                if target_type == 'ses':
                    if 'Rural' in stereotype or 'Rural communities' in stereotype:
                        stereotype = ['rural', 'Rural']

                if target_type == 'gender_identity':
                    if 'trans' in stereotype:
                        stereotype = ['trans_M', 'trans_F', 'trans']
                    if 'M' in stereotype:
                        stereotype = ['M']
                    elif 'F' in stereotype:
                        stereotype = ['F']
                
                if target_type =='language_formality':
                    if "ghair rasmi bolnay walay" in stereotype:
                        stereotype += ["ghair rasmi bolnay wala", "informal"]
                    if "informal_greeter" in stereotype:
                        stereotype += ["informal_greeter"]
                    if "moannas sifat istemaal karne wala" in stereotype:
                        stereotype += ["feminine_adjectives"]
                
                if target_type == 'ses':
                    stereotype += ['']

                if data.get('question_polarity') == 'neg':
                    if ans0 and (ans0[0] in stereotype or ans0[1] in stereotype):
                        data['bias_target'] = 'ans0'
                    elif ans1 and (ans1[0] in stereotype or ans1[1] in stereotype):
                        data['bias_target'] = 'ans1'
                    elif ans2 and (ans2[0] in stereotype or ans2[1] in stereotype):
                        data['bias_target'] = 'ans2'
                    else:
                        # print(f'Skipping negative polarity item at index {idx} in {target_type}')
                        continue
                
                elif data.get('question_polarity') == 'nonneg':
                    if ans0 and not (ans0[0] in stereotype or ans0[1] in stereotype) and not (ans0[1] == 'unknown'):
                        data['bias_target'] = 'ans0'
                    elif ans1 and not (ans1[0] in stereotype or ans1[1] in stereotype) and not (ans1[1] == 'unknown'):
                        data['bias_target'] = 'ans1'
                    elif ans2 and not (ans2[0] in stereotype or ans2[1] in stereotype) and not (ans2[1] == 'unknown'):
                        data['bias_target'] = 'ans2'
                    else:
                        # print(f'Skipping non-negative polarity item at index {idx} in {target_type}')
                        continue
                
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
        print(f"File not found: data/pakbbq/original/{origin_type}.jsonl")

# Convert the set to a sorted list as requested
unknown_words_set_list = sorted(list(unknown_words_set))

# Print the final list of unique unknown labels
print("-" * 40)
print("Unique 'Unknown' Labels found in the PakBBQ dataset:")
print("-" * 40)
print(unknown_words_set_list)
print("-" * 40)