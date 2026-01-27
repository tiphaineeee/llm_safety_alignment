# find_similar_principles.py

import numpy as np
import faiss
import json
import os
from tqdm import tqdm
import pickle

def load_principles_with_embeddings(filepath: str):
    """
    加载原则列表，假设格式：
    [
      {
        "principle": "You should be helpful.",
        "embedding": [0.1, 0.2, ..., 0.768]
      },
      ...
    ]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # data = json.load(f)
        data = [json.loads(line) for line in f]
    print(f"✅ 加载 {len(data)} 条原则")
    return data

def build_faiss_index(embeddings: np.ndarray, index_type="HNSW"):
    """
    构建 Faiss 索引
    :param embeddings: np.array of shape (N, D)
    :param index_type: "HNSW" or "IVFPQ"
    :return: faiss index
    """
    d = embeddings.shape[1]
    if index_type == "HNSW":
        # HNSW 索引（适合高精度近似最近邻）
        index = faiss.IndexHNSWFlat(d, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
    elif index_type == "IVFPQ":
        # IVFPQ 索引（适合超大规模）
        nlist = min(1000, len(embeddings) // 100)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)  # 8-bit PQ
        index.train(embeddings)
    else:
        raise ValueError("Unsupported index type")

    # 归一化向量（因为我们要用内积做相似度）
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"✅ 构建 Faiss 索引完成，类型: {index_type}")
    return index

def find_similar_pairs(principles, embeddings, thresholds=[0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65], top_k=5000):
    """
    查找所有相似度大于阈值的 pair
    使用 Faiss 进行近似最近邻搜索
    """
    n = len(principles)
    d = embeddings.shape[1]

    # 构建索引
    index = build_faiss_index(embeddings, index_type="HNSW")

    # 存储结果
    similar_pairs_by_threshold = {t: [] for t in thresholds}
    total_checked = 0

    # 逐个查询
    for i in tqdm(range(n), desc="🔍 查询相似对", unit="principle"):
        query_vec = embeddings[i].reshape(1, -1)

        # 搜索 top_k 最近邻（包括自己）
        distances, indices = index.search(query_vec, top_k)

        # 转换为相似度（内积 = 余弦相似度，因为已归一化）
        similarities = 1 - distances  # 因为 Faiss 返回的是距离（1 - cos_sim）

        # 遍历每个邻居
        for j, sim in zip(indices[0], similarities[0]):
            if i == j:  # 跳过自己
                continue
            if sim < 0.3:  # 提前剪枝，减少计算
                break
            total_checked += 1

            # 只记录 i < j 的 pair，避免重复
            if i < j:
                for th in thresholds:
                    if sim >= th:
                        similar_pairs_by_threshold[th].append({
                            'idx1': int(i),
                            'idx2': int(j),
                            'similarity': float(sim),
                            'principle1': principles[i]['prompt'],
                            'principle2': principles[j]['prompt']
                        })

    print(f"✅ 总共检查了 {total_checked} 对")
    return similar_pairs_by_threshold

def save_results(similar_pairs_by_threshold, output_dir="similar_pairs_30"):
    """
    保存结果到文件
    """
    os.makedirs(output_dir, exist_ok=True)

    for th, pairs in similar_pairs_by_threshold.items():
        filename = f"similar_pairs_{int(th*100)}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存 {len(pairs)} 对相似度 ≥{int(th*100)}% 的原则到: {filepath}")

    # 生成统计报告
    report = {
        "thresholds": {},
        "total_principles": len(similar_pairs_by_threshold[list(similar_pairs_by_threshold.keys())[0]][0]['principle1']) if similar_pairs_by_threshold else 0
    }
    for th, pairs in similar_pairs_by_threshold.items():
        report["thresholds"][str(th)] = {
            "count": len(pairs),
            "filename": f"similar_pairs_{int(th*100)}.json"
        }

    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📊 统计报告已保存至: {os.path.join(output_dir, 'summary.json')}")

# ===========================
# 🚀 主函数
# ===========================

def main():
    print("🚀 开始查找语义相似的原则对...")

    # Step 1: 加载数据
    input_file = "/mnt/oss_data/llm_safety/datasets/value_principles-embedding.jsonl"  # 替换为你自己的文件路径
    if not os.path.exists(input_file):
        print("❌ 文件不存在，正在生成示例数据...")

    principles = load_principles_with_embeddings(input_file)

    # Step 2: 提取 embedding
    embeddings = np.array([p['raw_output'] for p in principles], dtype=np.float32)
    print(f"🔢 embedding 维度: {embeddings.shape[1]}")

    # Step 3: 查找相似对
    thresholds = [0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
    similar_pairs = find_similar_pairs(principles, embeddings, thresholds=thresholds, top_k=5000)

    # Step 4: 保存结果
    save_results(similar_pairs)

    # Step 5: 输出统计
    print("\n📈 统计结果:")
    for th, pairs in similar_pairs.items():
        print(f"  ≥{int(th*100)}%: {len(pairs):,} 对")

    print("\n🎉 任务完成！")

if __name__ == "__main__":
    main()
