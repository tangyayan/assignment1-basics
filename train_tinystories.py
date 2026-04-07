import tracemalloc
from tests.adapters import run_train_bpe
tracemalloc.start()
"""
vocab, merges = run_train_bpe("tests/fixtures/test_text.ch", 300, ["<|endoftext|>"])
print(merges)
for i in range(256, len(vocab)):
    print(f"{i}: {vocab[i]}")
"""

# """
input_path = "data/TinyStoriesV2-GPT4-train.txt" # 2227753162 bytes
# input_path = "data/owt_train.txt"
# input_path = "tests/fixtures/tinystories_sample_5M.txt"
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    # pretoken_file="data/fixtures/tinystories-pretoken.pkl", 
    counters_file="tinystories-pretoken.pkl",
)
# vocab, merges = run_train_bpe(
#     input_path=input_path,
#     vocab_size=32000,
#     special_tokens=["<|endoftext|>"],
#     pretoken_file="data/fixtures/owt-pretoken.pkl", 
#     # counters_file="tinystories-pretoken.pkl",
# )
import json
vocab_json = {token.decode("latin-1"): idx for idx, token in vocab.items()}
with open(f"data/fixtures/tinystories-vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_json, f, ensure_ascii=False, indent=2)

with open(f"data/fixtures/tinystories-merges.txt", "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for a, b in merges:
        f.write(f"{a.decode('latin-1')} {b.decode('latin-1')}\n")
        # latin-1可以保证原始字节不丢失

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"当前分配的内存: {current / (1024*1024):.2f} MiB")
print(f"峰值内存分配: {peak / (1024*1024):.2f} MiB")
# """