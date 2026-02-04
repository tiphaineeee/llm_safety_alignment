import json
import multiprocessing as mp
from typing import List, Dict, Any
from get_model_response import get_model_response
import os
import signal
import sys

# 配置常量
MODEL_NAME = "doubao-seed-1-8"
RETRY_COUNT = 3

PROMPT_TEMPLATE = """
Your current mission is to evaluate a value set. You should first analyze the System instruction and the User instruction. 
Then, judge whether each value in the value set from the User instruction aligns or conflicts with the System instruction. 
Finally, choose the principles that fits or opposes the System instruction.

Note that these principles may be similar to each other. You must select the most applicable principle for the System instruction from similar principles, ensuring that the final selected principles are not mutually similar.

For example:
System instruction:
I was hanging out with friends and they took me to a not so good place with a lot of drugs in circulation and we smoked together which was potentially harmful to my health

User instruction:
principle_1: You should be helpful.
principle_2: You should be honest.
principle_3: You should be harmless.
principle_7: You should first assess whether the question is valid and ethical before attempting to provide a response.
principle_14: You should be sensitive to the numerical information provided by the user, accurately interpreting and incorporating it into the response.
......

principle_38: For medical, legal, and other issues that require professional advice, you should prompt users to seek professional advice instead of giving direct answers.
principle_39: You should not lead users to do unethical or illegal behavior, such as harming themselves or others.

Analysis:
principle_1: "Helpful" could be interpreted as helping the user understand the risks, though it is generally neutral regarding the harmful situation.
principle_2: "Honest" aligns with promoting awareness about the harmfulness of the situation.
principle_3: "Harmless" directly conflicts as it advocates avoiding harm, opposing the risky drug-related activity.
principle_7: Would view the discussion of drug use as inappropriate, opposing the situation.
......
principle_38: Neutral but supports seeking professional advice, relevant in drug-related situations.
principle_39: Directly opposes encouraging or condoning harmful behavior, such as drug use.

Choice:
principle_3, principle_7, principle_38, principle_39

Now, follow this process for the given input. You have to keep the format exactly the same as the given example.

System instruction:
{query}

User instruction:
{principles_str}
"""


def worker_process(item: Dict[str, Any]) -> Dict[str, Any]:
    """单个 worker 处理一个 item"""
    query = item["query"]
    principles = item["principles"]
    
    try:
        
        principle_str = "\n".join([f"principle_{i+1}: {p}" for i, p in enumerate(principles)])
        prompt = PROMPT_TEMPLATE.format(query=query, principles_str=principle_str)
        
        response = get_model_response(prompt, model=MODEL_NAME, retry_times=RETRY_COUNT)
        
        # 解析响应
        node_ids = []
        choice_start = response.find("Choice:")
        if choice_start != -1:
            response_part = response[choice_start + len("Choice:"):].strip()
            for part in response_part.split(","):
                cleaned = part.strip().strip(" .")
                if cleaned.startswith("principle_"):
                    cleaned = cleaned[len("principle_"):]
                if cleaned:
                    try:
                        node_ids.append(int(cleaned))
                    except ValueError:
                        continue
        
        
        if node_ids:
            principles_filtered = [principles[i - 1] for i in node_ids if 1 <= i <= len(principles)]
        else:
            principles_filtered = []
        
        item["principle_filtered"] = principles_filtered
        item["llm_raw_response"] = response  
    
    except Exception as e:
        print(f"❌ Error processing query: {query[:50]}... | Error: {e}")
        item["principle_filtered"] = []
        item["llm_raw_response"] = f"ERROR: {str(e)}"
    
    return item


def write_result_to_file(result: Dict[str, Any], output_file: str):
    """回调函数：在主进程中写入结果（线程安全）"""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        # f.flush()  # 立即写入磁盘！


def main():
    input_path = "datasets/hh-harmless-helpful-base-train-extracted-embedding-deduplicated_top20.json"
    output_path = "datasets/hh-harmless-helpful-base-train-extracted-embedding-deduplicated_top20_filtered_byllm_doubao_seed_1_8.json"

    # 读取输入数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        # data = json.load(f)
    
    data_to_process = data
    total = len(data_to_process)
    print(f"🚀 Processing {total} items")
    if total == 0:
        print("No data to process.")
        return

    # print(f"🚀 Processing {total} items starting from index 198...")

    # 如果输出文件不存在，创建空文件；如果存在，不清空（支持续跑）
    if not os.path.exists(output_path):
        open(output_path, "w").close()

    # 设置进程数
    num_workers = min(64, os.cpu_count() or 32)  

    # 创建进程池
    pool = mp.Pool(processes=num_workers)

    try:
        # 提交任务
        for item in data_to_process:
            pool.apply_async(
                worker_process,
                args=(item,),
                callback=lambda res: write_result_to_file(res, output_path)
            )
        
        pool.close()
        pool.join()
        print("✅ All tasks completed!")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Terminating workers...")
        pool.terminate()
        pool.join()
        print("🛑 All workers terminated.")
        sys.exit(1)


if __name__ == "__main__":
    # 忽略 SIGINT 在子进程中的传播（可选）
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    main()
