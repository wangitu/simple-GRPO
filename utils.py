import re
from tqdm import trange
from math_verify import parse, verify, ExprExtractionConfig

from transformers import GenerationConfig


def is_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) 
    if len(nums) == 0:
        return False
    
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return verify(ans, ground_truth)

def reward_correct(item, answer):
    return 1 if is_correct(item, answer) else -1

def reward_format(item, answer):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1


### constants
system_prompt = """You are a helpful assistant. A conversation occurs between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
###


def gen_answers(tokenizer, model, prompts):
    original_padding_side = tokenizer.padding_side
    original_pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}
            ], tokenize=False, add_generation_prompt=True)
        )
    
    answers = []
    for i in trange(0, len(tip_text), 16):
        inputs = tokenizer(tip_text[i: i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    
    tokenizer.padding_side = original_padding_side
    tokenizer.pad_token_id = original_pad_token_id
    
    return answers
