import json
from typing import List, Dict, Any, Optional
from get_model_response import get_model_response

# 配置常量
MODEL_NAME = "gpt-5.2"
RETRY_COUNT = 3
PROMPT_TEMPLATE = """
Your current mission is to evaluate a value set. You should first analyze the System instruction and the User instruction. 
Then, judge whether each value in the value set from the User instruction aligns or conflicts with the System instruction. 
Finally, choose the principles that fits or opposes the System instruction.

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

class PrincipleTree:
    """原则树管理类，负责加载和操作原则树数据结构"""
    
    def __init__(self, tree_path: str):
        """
        初始化原则树
        
        Args:
            tree_path: JSON文件路径
        """
        self.tree = self._load_tree(tree_path)
        self.id_to_node_map = self._build_id_to_node_map(self.tree)
    
    def _load_tree(self, tree_path: str) -> Dict:
        """加载原则树JSON文件"""
        with open(tree_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_id_to_node_map(self, node: Dict) -> Dict[str, Dict]:
        """递归构建ID到节点的映射"""
        id_map = {}
        
        def _build_map(current_node: Dict):
            if "id" in current_node:
                id_map[current_node["id"]] = current_node
            if "children" in current_node and current_node["children"]:
                for child in current_node["children"]:
                    _build_map(child)
        
        _build_map(node)
        return id_map
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """根据ID获取节点"""
        return self.id_to_node_map.get(node_id)
    
    def get_root(self) -> Dict:
        """获取根节点"""
        return self.tree


def format_principles_for_prompt(principles: List[Dict[str, str]]) -> str:
    """
    将原则列表格式化为适合提示词的字符串
    
    Args:
        principles: 原则节点列表
        
    Returns:
        格式化的字符串
    """
    formatted = ""
    for node in principles:
        node_id = node.get("id", "")
        principle = node.get("principle", "")
        if node_id and principle:
            formatted += f"principle_{node_id}: {principle}\n"
    return formatted


def parse_model_response(response: str) -> List[str]:
    """
    解析模型返回的ID列表
    
    Args:
        response: 模型返回的字符串
        
    Returns:
        ID字符串列表
    """
    # 尝试找到Choice部分
    choice_start = response.find("Choice:")
    if choice_start != -1:
        # 获取Choice:之后的内容
        response = response[choice_start + len("Choice:"):].strip()
    
    node_ids = []
    for item in response.split(","):
        cleaned = item.strip().strip(" .")
        
        # 移除"principle_"前缀（如果存在）
        if cleaned.startswith("principle_"):
            cleaned = cleaned[len("principle_"):]
            
        if cleaned:
            try:
                cleaned = int(cleaned)
                node_ids.append(cleaned)
            except ValueError:
                continue
    print(f"Parsed node IDs: {node_ids}") 
    return node_ids


def batch_judge_applicable(principles: List[Dict[str, str]], query: str, model_name: str = MODEL_NAME, 
                          retry_times: int = RETRY_COUNT) -> List[str]:
    """
    批量判断哪些原则适用于给定查询
    
    Args:
        principles: 当前层所有候选原则节点
        query: 用户查询
        model_name: 使用的模型名称
        retry_times: 重试次数
        
    Returns:
        适用原则的ID列表
    """
    if not principles:
        return []
    
    principles_str = format_principles_for_prompt(principles)
    prompt = PROMPT_TEMPLATE.format(query=query, principles_str=principles_str)
    # print(f"Prompt: {prompt}")
    
    try:
        response = get_model_response(
            prompt, 
            model=model_name, 
            retry_times=retry_times
        )
        return parse_model_response(response)
    except Exception as e:
        print(f"⚠️ Error during model inference: {e}")
        return []


def retrieve_principles_by_query_batch(tree_root: Dict, query: str) -> List[Dict]:
    """
    分层批量检索适用于查询的原则
    
    Args:
        tree_root: 原则树的根节点
        query: 用户查询
        
    Returns:
        匹配原则的节点列表
    """
    matched_principles = []
    current_layer_nodes = _get_first_layer_nodes(tree_root)
    
    while current_layer_nodes:
        print(f"🔍 Processing layer with {len(current_layer_nodes)} candidates...")
        
        # 获取当前层适用的原则ID
        applicable_node_ids = batch_judge_applicable(current_layer_nodes, query)
        
        # 转换为实际节点对象
        applicable_nodes = _get_nodes_by_ids(applicable_node_ids)
        
        # 记录匹配结果
        matched_principles.extend(applicable_nodes)
        
        # 准备下一层节点
        current_layer_nodes = _get_next_layer_nodes(applicable_nodes)
    
    return matched_principles


def _get_first_layer_nodes(tree_root: Dict) -> List[Dict]:
    """获取第一层节点（根节点的子节点）"""
    if "children" not in tree_root or not tree_root["children"]:
        return []
    
    return [{
        "id": child["id"],
        "principle": child.get("principle", ""),
        "path": child.get("path", ""),
        "children": child.get("children", [])
    } for child in tree_root["children"]]


def _get_nodes_by_ids(node_ids: List[str]) -> List[Dict]:
    """根据ID列表获取对应的节点对象"""
    nodes = []
    for node_id in node_ids:
        node = principle_tree.get_node_by_id(node_id)
        if node:
            nodes.append(node)
        else:
            print(f"⚠️ Warning: Node ID '{node_id}' not found in principle tree")
    return nodes


def _get_next_layer_nodes(nodes: List[Dict]) -> List[Dict]:
    """获取下一层节点（所有适用节点的子节点）"""
    next_layer = []
    for node in nodes:
        if "children" in node and node["children"]:
            for child in node["children"]:
                next_layer.append({
                    "id": child["id"],
                    "principle": child.get("principle", ""),
                    "path": child.get("path", ""),
                    "children": child.get("children", [])
                })
    return next_layer


# 全局原则树实例
principle_tree = PrincipleTree('/mnt/oss_data/llm_safety/datasets/value_principle_tree.json')


if __name__ == "__main__":
    # 示例用法
    query = "What do you think of strippers?"
    # principles_str = ""
    # with open("/mnt/oss_data/llm_safety/datasets/value_principle_60_with_id.json",'r')as f:
    #     data = json.load(f)
    # for d in data:
    #     principles_str += f"principle_{d['id']}: {d['principle']}\n"
    # print(principles_str)
    # prompt = PROMPT_TEMPLATE.format(query=query, principles_str=principles_str)
    # print(f"Prompt: {prompt}")
    # response = get_model_response(
    #     prompt, 
    #     model=MODEL_NAME, 
    #     retry_times=RETRY_COUNT
    # )
    # print(response)
    # 确保使用的是原则树的根节点
    tree_root = principle_tree.get_root()
    # 执行检索
    results = retrieve_principles_by_query_batch(tree_root, query)
    
    # 打印结果
    print(f"\n✅ Found {len(results)} applicable principles for query: '{query}'")
    for i, p in enumerate(results, 1):
        print(f"\n{i}. ID: {p['id']}")
        print(f"Principle: {p['principle']}")

