from huggingface_hub import hf_hub_download

# 下载单个文件
path = hf_hub_download(
    repo_id="stanford-cs336/owt-sample",
    filename="owt_train.txt.gz",
    repo_type="dataset",
    local_dir="."
)
print("Downloaded to:", path)
