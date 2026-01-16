import json
import re
def extract_rot_field_unique(input_file, output_file):
    """
    提取SOCIAL-CHEM-101准则的 rot 字段作为principle，并去重
    """
    seen_principles = set()
    count = 0
    duplicates = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line.strip())
                
                rot_value = data.get('rot')
                if rot_value and rot_value.strip():
                    # 检查是否已存在
                    if rot_value not in seen_principles:
                        seen_principles.add(rot_value)
                        new_data = {
                            'principle': rot_value,
                            'source': 'SOCIAL-CHEM-101'
                        }
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                        count += 1
                    else:
                        duplicates += 1
                
            except Exception as e:
                continue
    
    print(f"处理完成！")
    print(f"唯一 principles: {count} 条")
    print(f"重复: {duplicates} 条")
    return count

def process_hh_rlhf_data(input_file, output_file):
    """
    处理HH-RLHF数据，提取里面的query
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = [json.loads(line) for line in infile]
    queries = [item['chosen'] for item in data]
    pattern = r'Human:\s*(.*?)\s*\n\s*\n\s*Assistant:'
    extracted_queries = []
    for q in queries:
        match = re.search(pattern, q, re.DOTALL)
        if match:
            human_content = match.group(1).strip()
            extracted_queries.append(human_content)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for query in extracted_queries:
            outfile.write(json.dumps({"query":query},ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # extract_rot_field_unique(
    #     input_file='/home/luoshi6/safety_alignment/datas/social-chem-101.v1.0.jsonl',
    #     output_file='/home/luoshi6/safety_alignment/datas/social-chem-101-principles.jsonl'
    # )
    process_hh_rlhf_data(
        input_file='/home/luoshi6/safety_alignment/datas/hh-helpful-base-train.jsonl',
        output_file='/home/luoshi6/safety_alignment/datas/hh-helpful-base-train-extracted.jsonl'
    )
