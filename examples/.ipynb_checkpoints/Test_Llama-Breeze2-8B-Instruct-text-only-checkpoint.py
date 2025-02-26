from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import torch
from mtkresearch.llm.prompt import MRPromptV3

model_id = 'mradermacher/Llama-Breeze2-8B-Instruct-text-only-GGUF'
model_file = 'Llama-Breeze2-8B-Instruct-text-only.Q4_K_M.gguf'

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=model_file)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=model_file)

# model_id = 'MediaTek-Research/Llama-Breeze2-8B-Instruct-v0_1'
# model = AutoModel.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto',
#     img_context_token_id=128212
# ).eval()

# model = AutoModel.from_pretrained("mradermacher/Llama-Breeze2-8B-Instruct-text-only-GGUF")

# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

generation_config = GenerationConfig(
  max_new_tokens=2048,
  do_sample=True,
  temperature=0.01,
  top_p=0.01,
  repetition_penalty=1.1,
  eos_token_id=128009
)

prompt_engine = MRPromptV3()

sys_prompt = 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.'

def _inference(tokenizer, model, generation_config, prompt, pixel_values=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if pixel_values is None:
        output_tensors = model.generate(**inputs, generation_config=generation_config)
    else:
        output_tensors = model.generate(**inputs, generation_config=generation_config, pixel_values=pixel_values.to(model.dtype))
    output_str = tokenizer.decode(output_tensors[0])
    return output_str

conversations = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "請問什麼是深度學習？"},
]

prompt = prompt_engine.get_prompt(conversations)
output_str = _inference(tokenizer, model, generation_config, prompt)
result = prompt_engine.parse_generated_str(output_str)
print(result)
# {'role': 'assistant', 'content': '深度學習是一種人工智慧技術，主要是透過模仿生物神經網路的結構和功能來實現。它利用大量數據進行訓練，以建立複雜的模型並使其能夠自主學習、預測或分類輸入資料。\n\n在深度學習中，通常使用多層的神經網路，每一層都包含許多相互連接的節點（稱為神經元）。這些神經元可以處理不同特徵的輸入資料，並將結果傳遞給下一層的神經元。隨著資料流向更高層次，這個過程逐漸捕捉到更抽象的概念或模式。\n\n深度學習已被廣泛應用於各種領域，如圖像識別、自然語言處理、語音識別以及遊戲等。它提供了比傳統機器學習方法更好的表現，因為它能夠從複雜且非線性的數據中提取出有用的資訊。'}
