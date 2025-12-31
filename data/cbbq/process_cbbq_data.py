import json
import os
import re

def clean_dict_keys(data):
    """Recursively clean BOM and invisible characters from dict keys"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            clean_key = key.replace('\ufeff', '').replace('\u200b', '').strip()
            cleaned[clean_key] = clean_dict_keys(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_dict_keys(item) for item in data]
    else:
        return data


# Category mapping
def get_category_mapping():
    """Return mapping from CBBQ to BBQ category"""
    return {
        'race': 'race_ethnicity',
        'gender': 'gender_identity',
        'disability': 'disability_status',
        # Others remain unchanged
        'age': 'age',
        'religion': 'religion',
        'nationality': 'nationality',
        'SES': 'ses',
        'physical_appearance': 'physical_appearance',
        'sexual_orientation': 'sexual_orientation',
        'educational_qualification': 'educational_qualification',
        'household_registration': 'household_registration',
        'ethnicity': 'ethnicity',
        'disease': 'disease'
    }


def find_unknown_id(item):
    """Find the ID of the 'unknown' option, return string format 'ans0', 'ans1', 'ans2'"""
    unknown_keywords = ['不确定', '不知道', '无法判断', '无法确定', '未知', 
                       '说不准', '不清楚', '无从得知', '没有足够信息', 'unknown']
    
    for i in range(3):  
        ans_key = f'ans{i}'  
        if ans_key in item:  
            ans_text = item[ans_key].replace('人人', '人')
            ans_text_lower = ans_text.lower()
            
            for keyword in unknown_keywords:
                if keyword.lower() in ans_text_lower:
                    return f'ans{i}'  # Return string format
    return None


def get_bias_group_mappings():
    """Return mappings and synonym expansions for bias_targeted_groups"""
    mappings = {
        # ====================== Age Related ======================
        "00后/年轻人/10后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "00后/年轻人/11后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "00后/年轻人/12后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "00后/年轻人/13后": ["高中一年级", "高中生", "高中三年级", "大学生", "大学新生", "大学二年级"],
        "00后/年轻人/14后": ["高中一年级", "高中生", "高中三年级", "大学生", "大学新生", "大学二年级"],
        "00后/年轻人/15后": ["高中一年级", "高中生", "高中三年级", "大学生", "大学新生", "大学二年级"],
        "00后/年轻人/16后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "00后/年轻人/17后": ["高中一年级", "大学生", "高中生", "高中三年级", "大学新生", "大学二年级"],
        "00后/年轻人/18后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "00后/年轻人/19后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "00后/年轻人/20后": ["00后", "Z世代的孩子", "Z世代","千禧一代"],
        "年轻人": ["28岁的", "青年人", "高中生", "年轻的成年人人", "年轻的家伙", "25岁的", "24岁的人", "25岁的人人",
                  "少女人", "年轻的成年人", "24岁的", "青年", "二十多岁人", "22岁的人人", "24岁的人人", "28岁的人人",
                  "年轻的家伙人", "20岁的人", "男孩", "22岁的", "高中生人", "25岁的人", "20岁的", "少年", "二十多岁",
                  "22岁的人", "男孩人", "少年人", "少女"],
        "老年人": ["75岁的", "老奶奶", "65岁的人", "奶奶", "老年", "68岁的", "六十多岁的老人", "60岁的人",
                  "老爷爷", "六十多岁的老", "63岁的", "68岁的人", "75岁的人", "外婆", "65岁的", "70岁的",
                  "爷爷", "63岁的人", "老年人", "外公", "60岁的", "70岁的人","50岁的人","55岁的人","70岁的人","中年人"],
        "老龄人": ["65岁的人", "51岁的人", "48岁的", "55岁的", "中年", "外婆", "65岁的", "55岁的人人",
                  "40岁的人", "53岁的人", "45岁的", "老年人", "53岁的", "48岁的人人", "退休者", "老年",
                  "52岁的人人", "50岁的人人", "35岁的人人", "55岁的人", "退休", "51岁的", "60岁的人",
                  "54岁的人", "老前辈", "52岁的", "爷爷", "50岁的", "外公", "60岁的", "35岁的人", "35岁的",
                  "40岁的", "45岁的人人", "52岁的人", "中年人", "48岁的人", "50岁的人", "45岁的人", "40岁的人人",
                  "54岁的", "奶奶", "60岁的人人", "60岁的人人"],

        # ====================== Gender Related ======================
        "女性": ["女同学", "女教授", "女董事", "女性创业", "女CEO", "女博士", "女性CEO", "丧偶的女性",
                "女性顾客", "女创业", "女艺术家", "孩子的妈妈", "未婚女性", "女性艺术家", "女职员",
                "女生", "女性", "女性硕士毕业生", "离了婚的女性", "女性董事", "女性面试", "女创业者",
                "女员工", "女性面试者", "女性创业者", "女性朋友"],

        # ====================== Sexual Orientation Related ======================
        "非异性恋者": ["无性恋男", "同性恋男人", "同性恋", "无性恋", "泛性恋", "无性恋朋友", "双性恋",
                    "无性恋男人", "泛性恋男人", "无性恋者", "同性恋朋友", "同性恋男", "同性恋者",
                    "双性恋朋友", "双性恋者", "双性恋男人", "双性恋男", "泛性恋男", "泛性恋朋友", "泛性恋者"],
        "变性人": ["变性女孩", "变性女生"],

        # ====================== Race/Ethnicity/Nationality Related ======================
        "中国人": ["中国人"],
        "华人": ["华裔女人", "中国女人", "华裔男人", "华裔", "中国男", "华裔男", "中国", "中国女", "中国男人"],
        "俄罗斯人": ["俄罗斯人"],
        "美国人": ["美国学生", "美国人", "美国"],
        "英国人": ["英国", "英国人"],
        "荷兰人": ["荷兰", "荷兰人"],
        "西班牙人": ["西班牙", "西班牙人"],
        "韩国人": ["韩国人", "韩国"],
        "日本人/阿拉伯人/厄瓜多尔/保加利亚": ["保加利亚", "厄瓜多尔人", "日本", "阿拉伯人", "厄瓜多尔",
                                            "日本人", "保加利亚人", "阿拉伯"],
        "格鲁吉亚人": ["格鲁吉亚人", "格鲁吉亚"],
        "印度人/巴基斯坦/斯里兰卡人": ["巴基斯坦", "巴基斯坦人", "印度人", "印度", "斯里兰卡人", "斯里兰卡"],
        "越南人/泰国人/印度尼西亚人/缅甸人/日本人/韩国人/中国人/印度人/巴基斯坦人": [
            "韩国人", "越南", "中国人", "缅甸人", "巴基斯坦人", "印度人", "越南人", "印度尼西亚",
            "泰国人", "日本人", "泰国", "中国", "缅甸", "印度尼西亚人", "印度", "巴基斯坦", "韩国"],
        "阿富汗人/伊朗人/伊拉克人/利比亚人/摩洛哥人/巴勒斯坦人/沙特人/叙利亚人/也门人": [
            "沙特人", "阿富汗人", "阿拉伯", "阿富汗", "巴勒斯坦人", "沙特", "也门人", "伊朗人",
            "叙利亚", "摩洛哥人", "利比亚", "也门", "伊拉克人", "利比亚人", "阿拉伯人", "伊拉克",
            "叙利亚人", "巴勒斯坦", "摩洛哥", "伊朗"],
        "埃塞俄比亚人/科特迪瓦/肯尼亚人/莫桑比克人/尼日利亚人/塞内加尔": [
            "科特迪瓦人", "莫桑比克人", "塞内加尔", "肯尼亚", "塞内加尔人", "肯尼亚人", "埃塞俄比亚人",
            "尼日利亚人", "埃塞俄比亚", "尼日利亚", "莫桑比克", "科特迪瓦"],
        "黑种人/非裔美国人/拉美裔人/非洲人": ["黑皮肤的", "黑种人", "黑皮肤的学生", "黑皮肤的女人",
                                            "拉美裔美国女", "拉美裔美国男", "拉美裔美国学生", "黑种",
                                            "黑皮肤的男", "拉美裔美国", "黑皮肤", "黑皮肤的女", "来自中东地区的人",
                                            "黑皮肤的男人", "拉美裔美国女人", "拉美裔美国男人", "黑皮肤的男人",
                                            "非裔美国男人", "非裔美国女人", "非裔美国学生", "非裔美国人", "非洲人", 
                                            "拉美裔人","非洲男人","非洲女人","非洲","阿拉伯人","中东地区的人"],
        "华裔/韩裔/日裔/越南裔/犹太裔/泰裔/印度裔/菲裔/黄种人": [
            "欧洲学生", "泰裔", "犹太", "犹太裔", "拉美裔美国学生", "韩裔学生", "欧洲人", "韩裔",
            "犹太裔学生", "黑人", "欧洲", "华裔", "华裔学生", "菲裔学生", "西班牙裔美国学生",
            "西班牙裔美国", "美国白", "印度裔学生", "泰裔学生", "拉美裔美国", "日裔学生",
            "美国白人", "日裔", "犹太人", "犹太学生", "西班牙裔美国人", "印度裔", "菲裔",
            "越南裔", "白种美国学生", "越南裔学生", "拉美裔美国人"],
        "犹太人": ["犹太人", "西班牙裔美国", "美国白", "犹太", "拉美裔美国", "美国白人", "黑人", "亚裔", "华裔"],

        # ====================== Chinese Ethnic Minorities ======================
        "云南少数民族": ["云南少数民族", "云南少数民族人"],
        "维吾尔族": ["维吾尔族人", "维吾尔族"],
        "藏族": ["藏族", "藏族人"],
        "苗族": ["苗族人", "苗族"],
        "彝族": ["彝族", "彝族人"],
        "壮族": ["壮族", "壮族人"],
        "蒙古族": ["蒙古族人", "蒙古族"],

        # ====================== Religion Related ======================
        "佛教": ["信奉佛教的人", "佛教信徒", "信奉佛教的"],
        "基督教": ["信奉基督教的邻居", "信奉基督教的", "信奉基督教", "基督教信徒"],
        "道教": ["信奉道教的", "信奉道教的人", "道教信徒"],
        "印度教": ["印度教信徒"],
        "伊斯兰教/穆斯林": ["伊斯兰教信徒", "信奉伊斯兰教的的朋友", "穆斯林"],
        "无神论者": ["道教信徒", "无神论", "伊斯兰教信徒", "佛教信徒", "无神论者",
                    "犹太教信徒", "印度教信徒", "基督教信徒"],

        # ====================== Disability Related ======================
        "听力语言残疾/聋哑人/肢体残疾/意外事故截肢/肢残人/精神病残疾/视力残疾/盲人/多重残疾/综合残疾": [
            "肢残", "患有视力残疾的孩子", "因意外事故截肢的人", "聋哑", "残障", "患有视力残疾的人",
            "患有肢体残疾的孩子", "患有肢体残疾的人", "残疾", "患有多重残疾的人", "患有精神病残疾的人",
            "患有多重残疾的", "患有综合残疾的", "患有听力语言残疾的孩子", "患有肢体残疾的", "精神病残疾人士",
            "患有残疾的孩子", "患有精神病残疾的孩子", "聋哑人", "因意外事故截肢人", "患有多重残疾的孩子",
            "患有综合残疾的人", "肢残人士", "因意外事故截肢的", "患有听力语言残疾的人", "患有听力语言残疾的",
            "患有精神病残疾的", "精神病残疾", "患有视力残疾的", "患有残疾的", "听力语言残疾人士",
            "患有残疾的人", "残疾人", "因意外事故截肢", "听力语言残疾", "患有综合残疾的孩子", "残障人士"],

        # ====================== Disease Related ======================
        "新冠患者": ["传染病患者", "传染病患"],
        "乙肝患者": ["乙肝患者", "乙肝患"],
        "艾滋病患者": ["艾滋病患", "精神/心理疾病患", "精神/心理疾病患者", "艾滋病患者"],
        "抑郁症患者": ["心理疾病患", "小刘的朋友", "心理疾病患者"],
        "精神疾病患者": ["精神疾病患者", "精神疾病患", "精神/心理疾病患者", "精神/心理疾病患"],
        "精神病患者": ["精神疾病患者", "精神疾病患", "精神/心理疾病患者", "精神/心理疾病患"],

        # ====================== Appearance Related ======================
        "身高较矮者": ["个子很矮者", "个子很矮", "个子较矮者", "个子较矮"],
        "较为肥胖者": ["有点胖者", "有点胖", "较胖者", "较胖", "身材肥硕", "身材肥硕者","身材不太好者","非常胖者"],
        "外貌不出众者": ["欠缺气质者", "长相普通者", "长相普通", "长相一般者", "欠缺气质", "长相一般","不太打扮自己者"],

        # ====================== Economic/Household Registration/Education Related ======================
        "低收入人群/农民工群体": [
            "来自贫困阶层的家庭的孩子", "贫困阶层的群体", "贫困阶层的家庭的家长", "较贫穷的家长",
            "农村的家长", "温饱阶层的家庭的家长", "农民工家庭", "来自农民工家庭的家庭的孩子",
            "来自较贫穷的家庭", "较低收入的家庭的学生", "来自农民工家庭的家庭", "农民工",
            "农村的家庭的学生", "来自温饱阶层的家庭", "来自农村的家庭", "较贫穷的家庭的学生",
            "较低收入的家庭的朋友", "贫困阶层", "低产阶层", "来自贫困阶层的家庭", "较贫穷的家庭的朋友",
            "较低收入", "来自较低收入的家庭的孩子", "来自较低收入的家庭", "温饱阶层的群体",
            "低产阶层的家庭", "农村的家庭的朋友", "较贫穷", "来自较贫穷的家庭的孩子",
            "较低收入的家庭的家长", "温饱阶层", "来自低产阶层的家庭", "来自农村的家庭",
            "农民工群体", "农村的家庭的家长", "较低收入的选手", "来自低产阶层的家庭",
            "较低收入的家长", "农民工家庭的选手", "来自较低收入的家庭", "较低收入的群体",
            "低产阶层的家庭的家长", "低产阶层的家庭的学生", "较贫穷的群体", "低产阶层的朋友",
            "来自农村的家庭的孩子", "来自低产阶层的家庭的孩子", "较贫穷的家庭", "较低收入的家庭",
            "贫困阶层的家庭的学生", "温饱阶层的朋友", "温饱阶层的家庭", "贫困阶层的家庭的朋友",
            "农村的家庭", "贫困阶层的家长", "温饱阶层的家庭的朋友", "较贫穷的家庭的家长",
            "贫困阶层的家庭", "低产阶层的群体","工薪阶层的家庭的朋友","来自工薪阶层的的家庭的",
            "工薪阶层的家庭","工薪阶层的群体","来自农村的的家庭的","农村的的朋友","工薪阶层的家长","工薪阶层的的选手",
        ],
        "农名工/移民工": [
            "外地户籍者", "有农业户口的人", "有农业户口的", "外地农业户籍", "外地农民",
            "乡下的员工", "户籍登记地不在本地的人", "外地农业户籍员工", "乡下人", "外地员工",
            "外地工", "乡下", "有农业户籍的员工", "户籍登记地不在本地的", "有农业户籍的",
            "有农业户籍的人", "外地工人", "外地户籍员工", "外地农业户籍者", "移民工",
            "外地户籍", "外地", "户籍登记地不在本地的员工"
        ],
        "学历差异人群": ["低学历农民工", "低学历打工人", "毕业院校层次更低的毕业生", "低学历的",
                        "低学历打工", "低学历"],
        "第一学历差异人群": ["毕业院校层次更高的毕业生", "毕业院校层次更低的毕业生"],
        "非全日制类毕业生": ["全日制类毕业生"],
    }
    
    return mappings


def expand_bias_keywords(bias_groups):
    """Expand keywords based on bias_targeted_groups"""
    if not bias_groups:
        return []
    
    mappings = get_bias_group_mappings()
    keywords = []
    
    # Split original text
    groups = re.split(r'[/、，,]', bias_groups)
    
    for group in groups:
        group = group.strip()
        if not group:
            continue
        
        # Add original keyword
        keywords.append(group)
        
        # Find mapped synonyms
        for key, synonyms in mappings.items():
            if key in group or group in key:
                keywords.extend(synonyms)
        
        # Extract core word (remove suffix)
        core_word = group
        for suffix in ["人群", "群体", "者", "人士", "人员", "患者"]:
            if core_word.endswith(suffix):
                core_word = core_word[:-len(suffix)]
        if core_word and core_word != group and len(core_word) > 1:
            keywords.append(core_word)
    
    return list(set(keywords))  # Deduplicate


def find_bias_target(item):
    """
    Find the ID of the bias target option, return string format 'ans0', 'ans1', 'ans2'
    
    Core logic (consistent with original code):
    1. ambiguous + neg: bias_target = keyword option
    2. ambiguous + nonneg: bias_target = option that is not keyword and not unknown
    3. disambiguous + neg: bias_target = label (correct answer)
    4. disambiguous + nonneg: bias_target = option that is neither label nor unknown
    """
    bias_groups = item.get('bias_targeted_groups', '').strip()
    context_condition = item.get('context_condition', '').strip()
    question_polarity = item.get('question_polarity', '').strip()
    label = item.get('label')
    if label is not None:
        label = int(label)
    unknown_id = item.get('unknown_id')  # Now in string format 'ans0', 'ans1', 'ans2'
    
    if not bias_groups:
        return None
    
    # Convert unknown_id to integer for comparison (if exists)
    unknown_idx = None
    if unknown_id and isinstance(unknown_id, str) and unknown_id.startswith('ans'):
        unknown_idx = int(unknown_id[-1])
    elif isinstance(unknown_id, int):
        unknown_idx = unknown_id
    
    # Expand keywords
    bias_keywords = expand_bias_keywords(bias_groups)
    
    # Find option matching keywords (keyword_option)
    keyword_option = None
    max_matches = 0
    
    for i in range(3):
        ans_key = f'ans{i}'
        if ans_key not in item:
            continue
        
        ans_text = item[ans_key].replace('人人', '人')
        
        # Exclude unknown option
        if i == unknown_idx:
            continue
        
        # Count matching keywords
        matches = sum(1 for keyword in bias_keywords if keyword in ans_text)
        
        if matches > max_matches:
            max_matches = matches
            keyword_option = i
    
    # Determine bias_target based on different conditions
    if context_condition == 'ambiguous':
        if question_polarity == 'neg':
            return f'ans{keyword_option}' if keyword_option is not None else None  # Return string
        else:
            for i in range(3):
                if i != keyword_option and i != unknown_idx:
                    return f'ans{i}'  # Return string
    
    elif context_condition == 'disambiguous':
        if question_polarity == 'neg':
            return f'ans{label}' if label is not None else None  # Return string
        else:
            for i in range(3):
                if i != label and i != unknown_idx:
                    return f'ans{i}'  # Return string
    
    return f'ans{keyword_option}' if keyword_option is not None else None  # Return string


def add_target_fields(item, category):
    """Add unknown_id, bias_target and category to each data item"""
    # Clean all option texts
    for i in range(3):
        ans_key = f'ans{i}'
        if ans_key in item:
            item[ans_key] = item[ans_key].replace('人人', '人')
    
    # Ensure label is integer type
    if 'label' in item and item['label'] is not None:
        item['label'] = int(item['label'])
    
    # Add category field (using mapped name)
    category_mapping = get_category_mapping()
    item['category'] = category_mapping.get(category, category)

    context = item.get('context_condition', '').strip()
    if context == 'ambiguous':
        item['context_condition'] = 'ambig'
    elif context == 'disambiguous':
        item['context_condition'] = 'disambig'
    
    # Set unknown_id and bias_target (both in string format)
    item['unknown_id'] = find_unknown_id(item)
    item['bias_target'] = find_bias_target(item)
    return item


def merge_json_to_jsonl(ambiguous_path, disambiguous_path, output_jsonl_path, category):
    """Merge ambiguous and disambiguous JSON files into one JSONL file"""
    
    # Read ambiguous file
    if not os.path.exists(ambiguous_path):
        print(f"    Warning: ambiguous file does not exist")
        return None
    
    with open(ambiguous_path, 'r', encoding='utf-8-sig') as f:
        ambiguous_data = clean_dict_keys(json.load(f))
        print(f"    - ambiguous: {len(ambiguous_data)} items")
    
    # Read disambiguous file
    if not os.path.exists(disambiguous_path):
        print(f"    Warning: disambiguous file does not exist")
        return None
    
    with open(disambiguous_path, 'r', encoding='utf-8-sig') as f:
        disambiguous_data = clean_dict_keys(json.load(f))
        print(f"    - disambiguous: {len(disambiguous_data)} items")
    
    # Check if sample counts match
    if len(ambiguous_data) != len(disambiguous_data):
        print(f"    Error: Sample counts do not match! ambiguous={len(ambiguous_data)}, disambiguous={len(disambiguous_data)}")
    
    # Merge data and add fields (pass category)
    all_data = []
    all_data.extend([add_target_fields(item, category) for item in ambiguous_data])
    all_data.extend([add_target_fields(item, category) for item in disambiguous_data])
    
    # Count None values
    null_count = sum(1 for item in all_data if item.get('bias_target') is None)
    
    if null_count > 0:
        print(f"    Warning: Found {null_count} items with bias_target as None")
    
    # Write JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(all_data), len(ambiguous_data), len(disambiguous_data), null_count


def main():
    # Define list of files to process
    files_to_convert = ['age', 'religion', 'race', 'gender', 'nationality', 'SES',
                        'physical_appearance', 'disability', 'sexual_orientation',
                        'educational_qualification', 'household_registration', 
                        'ethnicity', 'disease']
    
    base_dir = "data/cbbq/original/"
    output_dir = "data/cbbq"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get mapping relationship
    category_mapping = get_category_mapping()
    
    print("Starting conversion and merging of Chinese BBQ dataset...")
    print("=" * 70)
    
    total_records = 0
    total_null_count = 0
    success_count = 0
    skipped_count = 0
    
    # For generating README
    stats = []
    
    for filename in files_to_convert:
        # Use mapped filename
        mapped_filename = category_mapping.get(filename, filename)
        print(f"\nProcessing {filename} -> {mapped_filename}:")
        
        ambiguous_file = os.path.join(base_dir, filename, "ambiguous", "ambiguous.json")
        disambiguous_file = os.path.join(base_dir, filename, "disambiguous", "disambiguous.json")
        # Output filename uses mapped name
        output_file = os.path.join(output_dir, f"{mapped_filename}.jsonl")
        
        if not os.path.exists(ambiguous_file) and not os.path.exists(disambiguous_file):
            print(f"  Error: Files for {filename} do not exist, skipping")
            skipped_count += 1
            continue
        
        try:
            # Pass original category name to add to samples
            result = merge_json_to_jsonl(ambiguous_file, disambiguous_file, output_file, filename)
            
            if result is None:
                skipped_count += 1
                if os.path.exists(output_file):
                    os.remove(output_file)
                continue
            
            num_records, amb_count, disamb_count, null_count = result
            total_records += num_records
            total_null_count += null_count
            success_count += 1
            
            # Record statistics
            stats.append({
                'category': mapped_filename,
                'total': num_records,
                'ambiguous': amb_count,
                'disambiguous': disamb_count
            })
            
            null_info = f" (including {null_count} null items)" if null_count > 0 else ""
            print(f"  Completed -> {mapped_filename}.jsonl ({num_records} items){null_info}")
        
        except Exception as e:
            print(f"  Error: Conversion failed for {filename}: {e}")
            skipped_count += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Conversion completed!")
    print(f"  Success: {success_count}/{len(files_to_convert)} categories")
    print(f"  Skipped: {skipped_count}/{len(files_to_convert)} categories")
    print(f"  Total: {total_records} data items")
    if total_null_count > 0:
        print(f"  Warning: Total {total_null_count} items have bias_target as None")
    print(f"  Output directory: {output_dir}")
    
    # Generate README.md
    generate_readme(output_dir, stats, total_records)
    print(f"\nREADME.md generated")


def generate_readme(output_dir, stats, total_records):
    """生成 README.md 文件"""
    readme_path = os.path.join(output_dir, "README.md")
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# Chinese BBQ Dataset Statistics\n\n")
        f.write("## Dataset Overview\n\n")
        f.write(f"Total samples: **{total_records:,}**\n\n")
        f.write("## Data Statistics by Category\n\n")
        f.write("| Category | Total | Ambiguous | Disambiguous |\n")
        f.write("|----------|-------|-----------|-------------|\n")
        
        for stat in stats:
            f.write(f"| {stat['category']} | {stat['total']:,} | {stat['ambiguous']:,} | {stat['disambiguous']:,} |\n")
        
        f.write("\n## Category Mapping (CBBQ → BBQ)\n\n")
        f.write("| CBBQ Category | BBQ Category |\n")
        f.write("|---------------|---------------|\n")
        
        category_mapping = get_category_mapping()
        for cbbq_cat, bbq_cat in sorted(category_mapping.items()):
            if cbbq_cat != bbq_cat:
                f.write(f"| {cbbq_cat} | {bbq_cat} |\n")
        
        f.write("\n## Files\n\n")
        for stat in stats:
            f.write(f"- `{stat['category']}.jsonl`\n")
        
        f.write("\n## Field Format\n\n")
        f.write("- `unknown_id`: String format ('ans0', 'ans1', 'ans2')\n")
        f.write("- `bias_target`: String format ('ans0', 'ans1', 'ans2')\n")
        f.write("- `label`: Integer format (0, 1, 2)\n")


if __name__ == "__main__":
    main()