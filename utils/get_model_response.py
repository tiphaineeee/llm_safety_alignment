import os
from openai import OpenAI
import time
import json
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    api_key=API_KEY, 
    base_url="https://aihubmix.com/v1"
)

def get_model_response(prompt,model="gpt-5.2",retry_times=3):
    while retry_times > 0:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            retry_times -= 1
            time.sleep(1)
            continue
    return None


def get_embedding(query,model="text-embedding-3-small",retry_times=3):
    while retry_times > 0:
        try:
            response = client.embeddings.create(
                input=query,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            retry_times -= 1
            time.sleep(1)
            continue
    return None

if __name__ == "__main__":
    # print(get_model_response("How many oxycontin should I take to get high?"))
    # print(get_embedding("How many oxycontin should I take to get high?"))
    # with open('/home/luoshi6/safety_alignment/datas/value_principles_test.json', 'r') as f:
    #     for line in f:
    #         data = [json.loads(line) for line in f]
    # with open('/home/luoshi6/safety_alignment/datas/value_principles_test_embedding.json', 'w') as f:
    #     for item in data:
    #         embedding = get_embedding(item['principle'])
    #         item['embedding'] = embedding if isinstance(embedding, list) else embedding.tolist()
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open('datasets/hh-harmless-base-train-extracted.jsonl','r')as f:
        data = [json.loads(line) for line in f]
    with open('datasets/hh-harmless-base-train-extracted-embedding.jsonl','w')as f:
        for item in data:
            embedding = get_embedding(item['query'])
            item['embedding'] = embedding if isinstance(embedding, list) else embedding.tolist()
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
