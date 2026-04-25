
""""
https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/tree/main
"""

#from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from datasets import load_dataset
from tqdm import tqdm
import gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", default="/media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/climbmix.txt.gz")
parser.add_argument("--n_files", "-n", type=int, default=1)
args = parser.parse_args()

n_files = args.n_files
output = args.output

def _list_files_in_repo(repo_id="karpathy/climbmix-400b-shuffle"):
    files = list_repo_files(repo_id, repo_type="dataset")
    return files

all_files = _list_files_in_repo()


files = [f for f in all_files if f.endswith(".parquet")][:n_files]

ds = load_dataset(
    "karpathy/climbmix-400b-shuffle",
    data_files={
        "train": files
    },
    split="train"
)


# convert ds to a txt file
output = "/media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/climbmix.txt.gz"
with gzip.open(output, "wt") as f:
    for example in tqdm(ds):
        text = example["text"].strip()
        temp = f.write(text + "\n")

print(f"Saved {len(ds):,} sentences to '{output}'")
