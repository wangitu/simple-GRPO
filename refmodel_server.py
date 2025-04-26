import os
import json
import queue
import bottle
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from grpo_args import args
from communication import (
    tensor_to_bytes,
    bytes_to_tensor,
    make_bytes_list,
    bytes_list_to_list,
)


def get_per_token_logps(ref_model, input_ids):
    logits = ref_model(input_ids).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps) # (B, L-1)


app = bottle.Bottle()
worker_queue = queue.LifoQueue()
result_queue = queue.LifoQueue()

@app.route('/upload', method='POST')
def do_upload():
    raw_data = bottle.request.body.read()
    try:
        raw_data = bytes_list_to_list(raw_data)
        data = {'base': json.loads(raw_data[0])} 
        data['inputs'] = bytes_to_tensor(raw_data[1])
        data['rewards'] = bytes_to_tensor(raw_data[2])
        data['gen_logps'] = bytes_to_tensor(raw_data[3])
        worker_queue.put(data)
        print('receive: ', data['inputs'].shape, data['rewards'], data['gen_logps'].shape)
        return b'success'
    except BaseException as e:
        err_msg = f'error: {e}'
        return err_msg.encode()

@app.route('/get', method='GET')
def do_get():
    if result_queue.empty():
        return b'empty'
    return result_queue.get()


def run_server():
    bottle.run(app, host='0.0.0.0', port=args.ref_server_port, server='tornado')


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa"
    ).to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    if args.ref_server_evaluation:
        import random
        from datasets import load_dataset, load_from_disk
        from utils import gen_answers, is_correct
        
        if not os.path.exists(args.data_path):
            train_dataset = load_dataset("openai/gsm8k", "main", split="train")
            train_dataset.save_to_disk(args.data_path)
        
        if not os.path.exists(args.data_path + '_test'):
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            dataset.save_to_disk(args.data_path + '_test')
        else:
            dataset = load_from_disk(args.data_path + '_test')
        QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
        random.seed(0)
        # samples = random.sample(QAs, 128)
        samples = QAs
        
        acc = 0
        answers = gen_answers(tokenizer, ref_model, [x["Q"] for x in samples])
        for i, (item, answer) in enumerate(zip(samples, answers)):
            acc += is_correct(item, answer)
        acc /= len(samples)
        print(f"Accuracy: {acc:.4f}")

    threading.Thread(target=run_server).start()
    
    while True:
        work_load = worker_queue.get()
        prompt_length = work_load['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(ref_model, work_load['inputs'].to(ref_model.device))
        per_token_logps = per_token_logps[:,prompt_length-1:]
        data = [
            json.dumps(work_load['base']).encode(), tensor_to_bytes(work_load['inputs']), tensor_to_bytes(work_load['rewards']),
            tensor_to_bytes(per_token_logps), tensor_to_bytes(work_load['gen_logps'])
        ]
        xdata = make_bytes_list(data)
        result_queue.put(xdata)
        