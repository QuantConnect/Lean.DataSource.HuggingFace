from CLRImports import *
from huggingface_hub import snapshot_download, hf_hub_url, get_hf_file_metadata
from os import environ, listdir, path, sep
from pathlib import Path
from shutil import  copyfile, rmtree

SUB_DIR = "models/huggingface/"
MODELS = [
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "yiyanghkust/finbert-tone",
    "ProsusAI/finbert",
    "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    "bardsai/finance-sentiment-fr-base",
    "nickmuchi/distilroberta-finetuned-financial-text-classification",
    "StephanAkkerman/FinTwitBERT-sentiment",
    "nickmuchi/sec-bert-finetuned-finance-classification",
    "nickmuchi/deberta-v3-base-finetuned-finance-text-classification",
    "google-bert/bert-base-uncased",
    "google/gemma-7b",
    "openai-community/gpt2",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert/distilbert-base-uncased",
    "FacebookAI/roberta-base",
    "microsoft/deberta-base"
]

def __get_destination(dst):
    if path.exists(dst):
        rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    return dst

def __get_commit_hash(repo_id):
    for filename in ["pytorch_model.bin", "tf_model.h5"]:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        try:
            metadata = get_hf_file_metadata(url=url)
            return metadata.commit_hash
        except:
            continue
    return ''

def __get_saved_hash(dir):
    file = dir / "commit_hash"
    if not path.exists(file):
        return ''
    with open(file) as f:
        return f.read()
    
def __save_current_hash(dir, src):
    file = dir / "commit_hash"
    with open(file, mode='w') as f:
        f.write(src.split(sep)[-1])

if __name__ == '__main__':
    token = environ.get("HF_ACCESS_TOKEN", None)
    if not token:
        exit(f'HF_ACCESS_TOKEN is not defined')

    root = Path(Config.Get("processed-data-directory", Globals.DataFolder)) / SUB_DIR
    temp = Path(Config.Get("temp-output-directory", "/temp-output-directory")) / SUB_DIR
    temp.mkdir(parents=True, exist_ok=True)

    errors = ''

    for repo_id in MODELS:
        current_hash = __get_commit_hash(repo_id)
        if not current_hash:
            errors += ' ' + repo_id
    
        saved_hash = __get_saved_hash(root / repo_id)
        if saved_hash and saved_hash == current_hash:
            continue

        src = snapshot_download(repo_id=repo_id, token=token, ignore_patterns=["*tfevents*"])
        dst = __get_destination(temp / repo_id)
        
        for file in listdir(src):
            copyfile(path.join(src, file), path.join(dst, file))
        __save_current_hash(dst, src)

    if errors:
        exit(errors)