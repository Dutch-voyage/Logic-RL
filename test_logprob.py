# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch

import transformers

from vllm import LLM, SamplingParams
from verl.utils.model import update_model_config
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from transformers import GenerationConfig

from verl.utils.torch_functional import pad_sequence_to_length


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize matrix of zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first column and first row of the matrix
    for i in range(m + 1):
        dp[i][0] = i  # Deletion from s1 to empty string
    for j in range(n + 1):
        dp[0][j] = j  # Insertion to s1 from empty string
    # Compute the Levenshtein distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # No cost if characters match
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]


def are_lists_similar(a, b):
    if len(a) != len(b):
        print("The lists are of different lengths.")
        return False

    total_length = 0
    total_diff = 0

    for s1, s2 in zip(a, b):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        diff = levenshtein(s1, s2)
        total_diff += diff
        print(f"Comparing strings:\n{s1}\n{s2}\nDifference: {diff} characters\n")

    percentage_difference = (total_diff / total_length) * 100
    print(f"Total difference: {percentage_difference:.2f}%")

    return percentage_difference <= 10


def test_vllm_spmd():
    assert torch.cuda.device_count() >= 2, 'At least 2 GPUs is required to run tp+dp tests.'

    # fill rollout config
    max_prompt_length = 200
    max_response_length = 2048

    local_model_path = "/home/yyx/RL/models/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    preencode_prompts = [
        f"<case_1> A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Zoey, and Ethan. Michael was heard saying, \"Ethan is a knight if and only if Michael is a knight\". \"Zoey is a knight or Ethan is a knight,\" Zoey mentioned. Ethan asserted: \"Michael is a knave if and only if Zoey is a knave\". So who is a knight and who is a knave?",
        f"<case_2> A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Zoey, and Ethan. Michael was heard saying, \"Ethan is a knight if and only if Michael is a knight\". \"Zoey is a knight or Ethan is a knight,\" Zoey mentioned. Ethan asserted: \"Michael is a knave if and only if Zoey is a knave\". So who is a knight and who is a knave?",
        f"<case_3> A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Zoey, and Ethan. Michael was heard saying, \"Ethan is a knight if and only if Michael is a knight\". \"Zoey is a knight or Ethan is a knight,\" Zoey mentioned. Ethan asserted: \"Michael is a knave if and only if Zoey is a knave\". So who is a knight and who is a knave?",
    ]
    temperature = 0.6
    top_p = 1
    
    kwargs = dict(n=1,
                  temperature=temperature,
                  top_p=top_p,
                  max_tokens=max_response_length,
                  logprobs=1,
                  ignore_eos=False)

    sampling_params = SamplingParams(**kwargs)

    llm = LLM(model=local_model_path,
              enable_sleep_mode=True,
              distributed_executor_backend="external_launcher",
              dtype='bfloat16',
              gpu_memory_utilization=0.5)

    print('start generation')

    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    
    for k, output in enumerate(outputs):
        vllm_response_tokens = []
        vllm_logprobs = []
        vllm_wordchoices = []
        prompt = output.prompt
        generated_text = output.outputs[0].text
        for logprob in output.outputs[0].logprobs:
            for key, value in logprob.items():
                vllm_logprobs.append(torch.tensor(value.logprob))
                vllm_wordchoices.append(torch.tensor(key))
        vllm_response_tokens.append(generated_text)
        print(f'vllm response: {vllm_response_tokens}')
        vllm_wordchoices = tokenizer.batch_decode(torch.stack(vllm_wordchoices, dim=0))
        import matplotlib.pyplot as plt
        vllm_logprobs = torch.stack(vllm_logprobs, dim=0)
        print(torch.exp(vllm_logprobs))
        chunk_size = 50
        for i in range(0, len(vllm_wordchoices), chunk_size):
            plt.figure(figsize=(20, 10))
            plt.plot(-vllm_logprobs[i:i+chunk_size] * torch.exp(vllm_logprobs[i:i+chunk_size]))
            plt.xticks(range(len(vllm_wordchoices[i:i+chunk_size])), vllm_wordchoices[i:i+chunk_size], rotation=45)
            if not os.path.exists(f"figures/{k}"):
                os.makedirs(f"figures/{k}")
            plt.savefig(f"figures/{k}/logprob_{i}.png")
            plt.close()
if __name__ == "__main__":
    test_vllm_spmd()
