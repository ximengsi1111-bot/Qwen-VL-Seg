import os
import torch

from vllm import LLM, SamplingParams


def main():
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_name", torch.cuda.get_device_name(0))

    model = os.environ.get("VLLM_SMOKE_MODEL", "/mingli01/models/Qwen3-VL-4B-Instruct")
    print("model", model, flush=True)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        trust_remote_code=True,
    )
    outs = llm.generate(
        ["Hello, reply with one word."],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    print("OUTPUT:", outs[0].outputs[0].text.strip(), flush=True)
    print("VLLM_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
