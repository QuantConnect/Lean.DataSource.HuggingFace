from CLRImports import *
from huggingface_hub import snapshot_download, hf_hub_url, get_hf_file_metadata
from os import environ, listdir, path, sep
from pathlib import Path
from shutil import move

SUB_DIR = "models/huggingface/"
MODELS = environ.get('MODELS').split(',')

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

        local_path = repo_id.replace('/', '--')
        saved_hash = __get_saved_hash(root / Path(f'models--{local_path}'))
        if saved_hash and saved_hash == current_hash:
            print(f'Skip updating \'{repo_id}\' already up to date, hash: {current_hash}')
            continue
        print(f'Start fetching \'{repo_id}\', current hash: {current_hash}, saved hash {saved_hash}...')

        src = snapshot_download(repo_id=repo_id, token=token, ignore_patterns=["*tfevents*"])
        # 'src' is: models--ProsusAI--finbert\snapshots\4556d13015211d73dccd3fdd39d39232506f3e43, we need to copy the whole folder for hugging to recognize it
        src = path.dirname(path.dirname(src))

        print(f'Downloaded: {src}')
        move(src, temp)
        __save_current_hash(temp / Path(path.basename(src)), current_hash)

    if errors:
        exit(errors)