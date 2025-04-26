import os
import re
import gc
import json
import time
import shutil
import random
import requests
import multiprocessing
from tqdm import tqdm, trange
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from grpo_args import GRPOArgs, args, setup_parallelism
from communication import (
    tensor_to_bytes,
    bytes_to_tensor,
    make_bytes_list,
    bytes_list_to_list,
)
from utils import system_prompt, is_correct, reward_correct, reward_format


ds_config = {
    "train_micro_batch_size_per_gpu": args.train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    }
}

engine = None
optimizer = None
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)


def get_rewards_logps():
    try:
        r = requests.get(f"{args.ref_server}/get").content
    except BaseException as e:
        print('get batch error: ', e)
        return None
    else:
        if r == b'empty':
            return None
        work_load = bytes_list_to_list(r)
        data = json.loads(work_load[0]) 
        data['inputs'] = bytes_to_tensor(work_load[1])
        data['rewards'] = bytes_to_tensor(work_load[2])
        data['ref_logps'] = bytes_to_tensor(work_load[3])
        data['gen_logps'] = bytes_to_tensor(work_load[4])
        return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['ref_logps'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    per_token_kl *= torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
    clipped_ratio = torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param)
    per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        
    per_token_loss = -(per_token_loss - args.beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


def gen_worker(conn: multiprocessing.connection.Connection, args: GRPOArgs):
    setup_parallelism()

    from vllm import LLM, SamplingParams
    
    vllm_gen = LLM(model=args.model_path, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(n=args.num_o_per_q, temperature=0.9, max_tokens=args.max_new_tokens)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
    
    try:
        dataset = load_from_disk(args.data_path)
    except BaseException as e:
        print(f'load dataset error: {e}')
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset.save_to_disk(args.data_path)
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    
    def gen_prompts(inputs):
        prompts = []
        for x in inputs:
            # the prompt with tokens that indicate the start of an assistant message will be appended to the formatted output
            prompts.append(tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["Q"]}
                ], tokenize=False, add_generation_prompt=True
            ))
        return prompts
    
    def gen_samples(inputs):
        prompt_texts = gen_prompts(inputs)
        print('Starting generation...')
        voutputs = vllm_gen.generate(prompt_texts, sampling_params, use_tqdm=False)
        
        answers = []
        ans_token_ids = []
        for v in voutputs:
            for out in v.outputs:
                answers.append(out.text)
                ans_token_ids.append(out.token_ids)
        
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i * args.num_o_per_q: (i + 1) * args.num_o_per_q]:
                rewards.append(reward_correct(inp, a) + reward_format(inp, a))

        return prompt_texts, torch.tensor(rewards, dtype=torch.float), answers, ans_token_ids
    
    def try_update_model():
        should_stop = False
        data = conn.recv()
        if data == 'stop':
            should_stop = True
        elif data == 'update':
            print('updating model')
            nonlocal vllm_gen
            del vllm_gen
            gc.collect()
            torch.cuda.empty_cache()
            
            # now rotate checkpoints
            ordering_and_checkpoint_paths = []
            glob_checkpoints = [str(x) for x in Path('./output').glob('step_*') if x.is_dir()]
            for path in glob_checkpoints:
                regex_match = re.match(f'.*step_(\d+)', path)
                if regex_match is not None:
                    ordering_and_checkpoint_paths.append((int(regex_match.group(1)), path))
            checkpoints_sorted = sorted(ordering_and_checkpoint_paths)
            checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
            latest_checkpoint = checkpoints_sorted.pop()
            for ckpt in checkpoints_sorted:
                print(f'removing {ckpt} due to total limit')
                shutil.rmtree(ckpt, ignore_errors=True)
            
            # now load the latest checkpoint
            print(f'loading {latest_checkpoint}')
            vllm_gen = LLM(model=latest_checkpoint, gpu_memory_utilization=0.95)
            
            # evaluate the latest checkpoint
            eval_params = SamplingParams(temperature=0, top_k=1, max_tokens=512)
            try:
                test_samples = load_from_disk(args.data_path + '_test')
            except BaseException as e:
                print(f'load test dataset error: {e}')
                test_samples = load_dataset("openai/gsm8k", "main", split="test")
                test_samples.save_to_disk(args.data_path + '_test')
            QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(test_samples['question'], test_samples['answer'])]
            random.seed(0)
            
            acc = 0
            test_samples = random.sample(QAs, 128)
            for i in trange(0, len(test_samples), args.eval_batch_size, desc="Evaluating with vllm..."):
                inputs = test_samples[i: i + args.eval_batch_size]
                prompts = gen_prompts(inputs)
                voutputs = vllm_gen.generate(prompts, eval_params, use_tqdm=False)
                answers = []
                for v in voutputs:
                    answers.append(v.outputs[0].text)
                for i, (item, answer) in enumerate(zip(inputs, answers)):
                    acc += is_correct(item, answer)
            acc /= len(test_samples)
            print(f"Accuracy: {acc:.4f}")
        
        return should_stop
                
            
    batches = 0
    num_batches_upon_update = args.world_size * args.update_steps
    while True:
        should_stop = False
        if batches >= num_batches_upon_update:
            print('waiting for update...')
            should_stop = try_update_model()
            batches = 0
            
        if conn.poll():
            data = conn.recv()
            if data == 'stop':
                should_stop = True
        
        if should_stop:
            print('vllm generation completed.')
            break
        
        inputs = random.sample(QAs, args.q_batch_size)
        prompt_texts, rewards, answers, ans_token_ids = gen_samples(inputs)
        for i, prompt in enumerate(prompt_texts):
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            plen = prompt_ids.shape[1]
            curr_rewards = rewards[i * args.num_o_per_q: (i + 1) * args.num_o_per_q]
            curr_ans_ids = ans_token_ids[i * args.num_o_per_q: (i + 1) * args.num_o_per_q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4:
                continue
            
            # standardize the rewards
            curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
            for j in range(0, args.num_o_per_q, args.train_batch_size):
                sub_rewards = curr_rewards[j:j + args.train_batch_size]
                sub_ans_ids = curr_ans_ids[j:j + args.train_batch_size]
                sub_ans_tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                output_ids = pad_sequence(sub_ans_tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                state_ids = torch.cat([
                    prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen), output_ids
                ], dim=-1) # (B, Lq + Lo)
                
                data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(state_ids), tensor_to_bytes(sub_rewards)]
                try: # may generate token that is out of vocabulary
                    logps = vllm_gen.generate(prompt_token_ids=state_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                    output_logps = [x.prompt_logprobs[plen:] for x in logps]
                    output_logps = torch.tensor([[list(x.values())[0].logprob for x in o] for o in output_logps]) # (B, Lo)
                    data.append(tensor_to_bytes(output_logps))
                    requests.post(f"{args.ref_server}/upload", data=make_bytes_list(data))
                    batches += 1
                except BaseException as e:
                    print(f"an error occurred: {e} while generating logps for pi_ref")
    
    conn.close()
            

if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()
    args.local_rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    if args.local_rank == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        recv_conn, send_conn = mp.Pipe(duplex=False)
        p = mp.Process(target=gen_worker, args=(recv_conn, args))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config, model=model, model_parameters=model.parameters()
    )

    progress = range(1, args.training_steps + 1)
    if args.local_rank == 0:
        progress = tqdm(progress)
    
    for step in progress:
        batch = get_rewards_logps()
        while batch is None:
            if args.local_rank == 0:
                print('waiting for batch...')
            time.sleep(1)
            batch = get_rewards_logps()

        loss = GRPO_step(batch)
        engine.backward(loss)
        engine.step()
        
        if args.local_rank == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
        
        if step % args.update_steps == 0:
            dist.barrier()
            if args.local_rank == 0:
                print('saving model')
                save_name = f"./output/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
                send_conn.send('update')
            dist.barrier()
            
    dist.barrier()
    if args.local_rank == 0:
        send_conn.send('stop')
        send_conn.close()
        p.join()
        print('Training completed.')
    