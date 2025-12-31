import json

dict_mapping = {
    'Age': 'age',
    'Disability_status': 'disability_status',
    'Gender_identity': 'gender_identity',
    'Physical_appearance': 'physical_appearance',
    "Sexual_orientation": "sexual_orientation"
}

# Collect all unknown texts (deduplicated)
all_unknown_texts = set()
unknown_by_category = {}  # Statistics by category

for origin_type in ['Gender_identity', 'Physical_appearance', "Sexual_orientation", "Age", "Disability_status"]:
    type_name = dict_mapping[origin_type]
    new_datas = []
    
    # Unknown texts for current category
    category_unknown_texts = set()
    
    with open(f'data/jpbbq/original/{origin_type}.jsonl', 'r', encoding='utf-8') as f, \
         open(f'data/jpbbq/{type_name}.jsonl', 'w', encoding='utf-8') as of:
        
        datas = [json.loads(line) for line in f if line.startswith('{')]
        
        for idx, data in enumerate(datas[:]):
            answer_infos = data['answer_info']
            ans0 = answer_infos['ans0']
            ans1 = answer_infos['ans1']
            ans2 = answer_infos['ans2']
            stereotype = data["additional_metadata"]['stereotyped_groups']

            if type_name == 'gender_identity':
                if 'trans' in stereotype:
                    stereotype = ['trans_M', 'trans_F', 'trans']
                if 'M' in stereotype:
                    stereotype = ['男性', 'm', '男の子', "夫", "M"]
                elif 'F' in stereotype:
                    stereotype = ['女性', 'f', '女の子', "妻", "F"]
            
            if type_name == 'disability_status':
                stereotype += ["全盲の人", "弱視の人"]
            
            if data['question_polarity'] == 'neg':
                if ans0[0] in stereotype or ans0[1] in stereotype:
                    data['bias_target'] = 'ans0'
                elif ans1[0] in stereotype or ans1[1] in stereotype:
                    data['bias_target'] = 'ans1'
                elif ans2[0] in stereotype or ans2[1] in stereotype:
                    data['bias_target'] = 'ans2'
                else:
                    print(f'dataset: {type_name}')
                    print(idx)
            
            elif data['question_polarity'] == 'nonneg':
                if not (ans0[0] in stereotype or ans0[1] in stereotype) and not (ans0[0] == 'unknown' or ans0[1] == 'unknown'):
                    data['bias_target'] = 'ans0'
                elif not (ans1[0] in stereotype or ans1[1] in stereotype) and not (ans1[0] == 'unknown' or ans1[1] == 'unknown'):
                    data['bias_target'] = 'ans1'
                elif not (ans2[0] in stereotype or ans2[1] in stereotype) and not (ans2[0] == 'unknown' or ans2[1] == 'unknown'):
                    data['bias_target'] = 'ans2'
                else:
                    print(f'dataset: {type_name}')
                    print(idx)
            
            # Find unknown_id and collect corresponding text
            unknown_text = None
            if 'unknown' in ans0[1]:
                data['unknown_id'] = 'ans0'
                unknown_text = ans0[0]
            elif 'unknown' in ans1[1]:
                data['unknown_id'] = 'ans1'
                unknown_text = ans1[0]
            elif 'unknown' in ans2[1]:
                data['unknown_id'] = 'ans2'
                unknown_text = ans2[0]
            
            # Add to set
            if unknown_text:
                all_unknown_texts.add(unknown_text)
                category_unknown_texts.add(unknown_text)
            
            if data['context_condition'] == 'disambig':
                if data['question_polarity'] == 'neg':
                    data['bias_target'] = 'ans' + str(data['label'])
                elif data['question_polarity'] == 'nonneg':
                    choices = ['ans0', 'ans1', 'ans2']
                    bias = 'ans' + str(data['label'])
                    choices.remove(bias)
                    choices.remove(data['unknown_id'])
                    data['bias_target'] = choices[0]
            
            data['category'] = type_name
            new_datas.append(data)
        
        for new_data in new_datas:
            of.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            of.flush()
        
        of.close()
    
    # Save unknown texts for current category
    unknown_by_category[type_name] = category_unknown_texts
    print(f"\nProcessing completed: {origin_type} -> {type_name}.jsonl")

# Print all deduplicated unknown texts
print("\n" + "="*80)
print("All Unknown Label Texts (Deduplicated):")
print("="*80)
for idx, text in enumerate(sorted(all_unknown_texts), 1):
    print(f"{idx:2d}. {text}")

print(f"\nTotal: {len(all_unknown_texts)} different unknown texts")

# Print unknown texts by category
print("\n" + "="*80)
print("Unknown Labels by Category:")
print("="*80)
for category, texts in unknown_by_category.items():
    print(f"\n{category}:")
    for text in sorted(texts):
        print(f"  - {text}")
    print(f"  Total: {len(texts)} types")