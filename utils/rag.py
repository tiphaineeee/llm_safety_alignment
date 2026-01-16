import json
import numpy as np
import faiss
import pickle
import os
from get_model_response import get_embedding
class PrincipleRetrievalSystem:
    def __init__(self, principles_json_path, index_path=None):
        self.principles = []
        self.embeddings = []
        self.index = None
        
        # 尝试加载已保存的索引
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)
        else:
            self._load_principles(principles_json_path)
            self._build_index()
            if index_path:
                self._save_index(index_path)
    
    def _load_principles(self, json_path):
        """从 JSON 文件加载 principles 和 embeddings"""
        with open(json_path, 'r', encoding='utf-8') as f:
             data = [json.loads(line) for line in f]
        print(len(data))
        if isinstance(data, dict):
            principles_list = data.get('principles', data.get('data', []))
        else:
            principles_list = data
        
        for item in principles_list:
            principle = item.get('principle')
            embedding = item.get('embedding')
            
            if principle and embedding:
                self.principles.append(principle)
                self.embeddings.append(embedding)
        
        print(f"加载了 {len(self.principles)} 条 principles")
    
    def _build_index(self):
        """使用 FAISS 构建索引"""
        embeddings_array = np.array(self.embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # 根据数据规模选择索引类型
        if len(self.embeddings) < 10000:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            # 使用 HNSW 进行近似搜索
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 200
        
        self.index.add(embeddings_array)
        print(f"FAISS 索引构建完成，维度: {dimension}, 向量数: {self.index.ntotal}")
    
    def _save_index(self, index_path):
        """保存索引和 principles"""
        faiss.write_index(self.index, f"{index_path}.faiss")
        with open(f"{index_path}.principles", 'wb') as f:
            pickle.dump(self.principles, f)
        print(f"索引已保存到 {index_path}")
    
    def _load_index(self, index_path):
        """加载已保存的索引"""
        self.index = faiss.read_index(f"{index_path}.faiss")
        with open(f"{index_path}.principles", 'rb') as f:
            self.principles = pickle.load(f)
        print(f"索引已加载，包含 {len(self.principles)} 条 principles")
    
    def get_query_embedding(self, query, model="text-embedding-3-small"):
        """获取 query 的 embedding 向量"""
        embedding = get_embedding(query, model)
        if isinstance(embedding, list):
            embedding = np.array(embedding).astype('float32')
        else:
            embedding = embedding.astype('float32')
        return embedding.reshape(1, -1)
    
    def retrieve(self, query, top_k=5):
        """检索最相关的 principles"""
        query_embedding = self.get_query_embedding(query)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.principles):
                results.append({
                    'principle': self.principles[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx)
                })
        
        return results

if __name__ == "__main__":
    rag = PrincipleRetrievalSystem(
        principles_json_path='/home/luoshi6/safety_alignment/datas/value_principles_test_embedding.json',
        index_path='/home/luoshi6/safety_alignment/datas/value_principles_test_embedding-principles.index'
    )
    print(rag.retrieve("How many oxycontin should I take to get high?"))