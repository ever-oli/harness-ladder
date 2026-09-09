import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openbmb/MiniCPM5-2B"
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
dtype = torch.float16 if use_cuda else torch.float32
print(f"Loading {model_id} on {device} with {dtype}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto" if use_cuda else None,
    trust_remote_code=True,
)
if not use_cuda:
    model = model.to(device)
inputs = tokenizer("Say hello in one sentence.", return_tensors="pt")
inputs.pop("token_type_ids", None)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
text = tokenizer.decode(output[0], skip_special_tokens=True)
print("OUTPUT:", text, flush=True)
if use_cuda:
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("ALLOCATED_VRAM_BYTES:", torch.cuda.memory_allocated(0), flush=True)
else:
    print("GPU: cpu", flush=True)
