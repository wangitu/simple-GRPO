import os
import torch
from dataclasses import dataclass


@dataclass
class GRPOArgs:
    model_path: str = '/data/wangqianle/models/qwen2.5-3b'
    data_path: str = './gsm8k'
    gen_device: int = 1
    beta: float = 0.04
    training_steps: int = 1000
    q_batch_size: int = 4
    train_batch_size: int = 1
    num_o_per_q: int = 4
    max_new_tokens: int = 512
    update_steps: int = 16
    clip_param: float = 0.2
    ref_server_port: int = 59875
    ref_server_evaluation: bool = True
    local_rank: bool = -1 # automatically set
    world_size: int = 1 # automatically set
    
    def __post_init__(self):
        self.eval_batch_size = self.q_batch_size * self.num_o_per_q
        self.ref_server = 'http://localhost:{}'.format(self.ref_server_port)
    

### constants
args = GRPOArgs()


def setup_parallelism():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gen_device)
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {args.gen_device}")
    