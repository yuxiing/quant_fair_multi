import json

for cat in ['gender','profession','religion','race']:
    file=open(f'data/stereoset_yuxing/{cat}_inter.jsonl','r',encoding='utf-8')
    if cat=='gender':
        cat='gender_identity'
    if cat=='race':
        cat='race_ethnicity'
    out_file=open(f'data/stereoset/{cat}.jsonl','w',encoding='utf-8')
    datas=[json.loads(line) for line in file if line.strip() and line.startswith('{')]
    print(f"Processing category: {cat} (file: {cat}_inter.jsonl)")
    for data in datas:
        data['bias_target'] = f"ans{str(data['biased'])}"
        data['answer']=f"ans{str(data['label'])}"
        data['category']=cat
        data['context_condition']='ambig'
        out_file.write(json.dumps(data,ensure_ascii=False)+'\n')
    file.close()
    out_file.close()
print("Processing completed!")
