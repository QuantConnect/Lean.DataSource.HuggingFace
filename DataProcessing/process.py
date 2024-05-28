from CLRImports import *
from huggingface_hub import snapshot_download, hf_hub_url, get_hf_file_metadata
from os import environ, listdir, path, sep
from pathlib import Path
from shutil import  copytree, rmtree

SUB_DIR = "models/huggingface/"
MODELS = environ.get('MODELS').split(',')

def __get_destination(dst):
    if path.exists(dst):
        rmtree(dst)
    return dst

def __get_commit_hash(repo_id):
    commit_hash = ''
    for filename in ["pytorch_model.bin", "tf_model.h5", "model.safetensors"]:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        try:
            metadata = get_hf_file_metadata(url=url)
            # we merge the bin & h5 hashes
            commit_hash = commit_hash + f'{filename}:{metadata.commit_hash}.'
        except:
            # might not exist
            continue
    return commit_hash

def __get_saved_hash(dir):
    file = dir / "commit_hash"
    if not path.exists(file):
        return ''
    with open(file) as f:
        return f.read()
    
def __save_current_hash(dir, current_hash):
    file = dir / "commit_hash"
    with open(file, mode='w') as f:
        f.write(current_hash)

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
            print(f'Skip updating \'{repo_id}\' already up to date, hash: {current_hash}')
            continue
        print(f'Start fetching \'{repo_id}\', hash: {current_hash}...')

        src = snapshot_download(repo_id=repo_id, token=token, ignore_patterns=["*tfevents*"])
        dst = __get_destination(temp / repo_id)

        copytree(src, dst)
        __save_current_hash(dst, current_hash)

    if errors:
        exit(errors)