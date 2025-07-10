# Portions copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import time
import json
import pathlib

import torch
import fairscale
import fire

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def load(ckpt_dir: str, tokenizer_path: str) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(pathlib.Path(ckpt_dir).glob("*.pth"))

    # For 7B model, there should be only one checkpoint file.
    assert len(checkpoints) == 1, f"Expected 1 checkpoint for 7B model, found {len(checkpoints)}"
    ckpt_path = checkpoints[0]

    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True)

    with open(pathlib.Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False, assign=True)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds.")
    return generator

def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.8, top_p: float = 0.95):
    torch.set_default_device('cpu')

    # Use float16 to save memory for non-mmapped tensors.
    torch.set_default_dtype(torch.float16)

    # LLaMA was designed for distributed training, so we need to "trick" it into thinking
    # it's running in a distributed environment with just one participant (ourselves).
    torch.distributed.init_process_group(
        backend="gloo",  # Communication protocol - gloo works on CPU, nccl only works on GPU
        init_method="tcp://127.0.0.1:23456",  # How processes find each other - TCP server on localhost
        rank=0,  # This process's ID (0 = first/only process)
        world_size=1  # Total number of processes (just 1 for single-process inference)
    )

    # Tell fairscale to split the model across 1 process (i.e., don't split it).
    fairscale.nn.model_parallel.initialize.initialize_model_parallel(1)

    # Ensure reproducible random number generation.
    torch.manual_seed(1)

    generator = load(ckpt_dir, tokenizer_path)
    prompts = ["The capital of Germany is the city of"]

    for i, prompt in enumerate(prompts):
        print(f"\n=== Prompt {i+1} ===")
        print(prompt, end="", flush=True)

        for token in generator.stream_generate(prompt, max_gen_len=256, temperature=temperature, top_p=top_p):
            print(token, end="", flush=True)

        print("\n" + "="*50)

if __name__ == "__main__":
    fire.Fire(main)
