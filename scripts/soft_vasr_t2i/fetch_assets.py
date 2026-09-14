"""Restore only SD1.5/IR assets with hf CLI; verify against PREVIOUS hashes."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/home/nguyenngocminh2005bn/soft-vasr-t2i-20260905')
cache = Path('/dev/shm/soft-vasr-t2i-20260905/cache')
hf = '/home/nguyenngocminh2005bn/flash-bon-tpu-20260904/venvs/main/bin/hf'
old = Path('/home/nguyenngocminh2005bn/dcs-table12-repro-20260903/results/table12-tpu-20260903')
manifest = json.loads((old/'asset_manifest.json').read_text())
selected = [f for f in manifest['files'] if
            ('models--runwayml--stable-diffusion-v1-5/snapshots/' in f['path']) or
            ('models--bert-base-uncased/snapshots/' in f['path']) or f['path'].startswith('ImageReward/')]
for repo, revision, subdir, needle in [
    ('runwayml/stable-diffusion-v1-5','451f4fe16113bff5a5d2269ed5ad43b0592e9a14','huggingface','models--runwayml--stable-diffusion-v1-5'),
    ('bert-base-uncased','86b5e0934494bd15c9632b12f734a8a67f723594','huggingface/transformers','models--bert-base-uncased'),
]:
    names = [f['path'].split(f'/snapshots/{revision}/')[1] for f in selected if needle in f['path']]
    subprocess.run([hf,'download',repo,*names,'--revision',revision,'--cache-dir',str(cache/subdir)],check=True)
subprocess.run([hf,'download','THUDM/ImageReward','ImageReward.pt','med_config.json',
                '--local-dir',str(cache/'ImageReward')],check=True)
# The unchanged BLIP loader requests the tokenizer's main alias. Resolve it and
# verify it points to the historical revision before enabling offline science.
subprocess.run([hf,'download','bert-base-uncased','config.json','tokenizer.json','tokenizer_config.json','vocab.txt',
                '--revision','main','--cache-dir',str(cache/'huggingface/transformers')],check=True)
assert (cache/'huggingface/transformers/models--bert-base-uncased/refs/main').read_text().strip() == '86b5e0934494bd15c9632b12f734a8a67f723594'
for entry in selected:
    path = cache/entry['path']
    with path.open('rb') as handle:
        actual = hashlib.file_digest(handle,'sha256').hexdigest()
    assert actual == entry['sha256'], f'Historical asset mismatch: {path}'
    assert path.stat().st_size == entry['bytes']
output = root/'results/restored_assets.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'all_match_historical_hashes':True,'files':selected},indent=2)+'\n')
print('ASSETS_VERIFIED',len(selected),flush=True)
