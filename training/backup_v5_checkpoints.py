import hashlib
import json
import os
import time
from pathlib import Path

import wandb

env = {}
for line in Path('/home/vivi/pixelated/.env').read_text().splitlines():
    s = line.strip()
    if s and not s.startswith('#') and '=' in s:
        k, _, value = s.partition('=')
        env[k.strip()] = value.strip().strip('"').strip("'")

ENTITY, PROJECT = 'wutang', 'pixelated-empathy-kan28'
NAME = 'arc-sft-baseline-d20b7a4d5678-sft-resume'
OUT = Path('/home/vivi/pixelated/ai/training/output/wandb_local_checkpoints_v5_20261020')
POLL = 60


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


api = wandb.Api(api_key=env['WANDB_API_KEY'])
seen: set[int] = set()
while True:
    try:
        coll = next((c for c in api.artifact_collections(f'{ENTITY}/{PROJECT}', 'lora', per_page=500) if c.name == NAME), None)
        if coll is not None:
            arts = sorted(coll.artifacts(per_page=300), key=lambda a: int(a.version.removeprefix('v')))
            for a in arts:
                v = int(a.version.removeprefix('v'))
                if v in seen:
                    continue
                dest = OUT / f'v{v}'
                if not (dest / '.mirrored').exists():
                    a.download(root=str(dest))
                    big = dest / 'adapter_model.safetensors'
                    digest = sha256_file(big) if big.exists() else None
                    (dest / '.mirrored').write_text(json.dumps({'version': v, 'sha256_adapter': digest, 'updated_at': str(a.updated_at)}) + '\n')
                    os.chmod(dest / '.mirrored', 0o600)
                seen.add(v)
                print('mirrored', v, flush=True)
    except Exception as e:
        print('ERR', repr(e), flush=True)
    time.sleep(POLL)
