"""Install optional, pinned CPU models without changing the workbench environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'work' / 'pseudo_models'
REPOS = {
    'Depth-Anything-V2': ('DepthAnything/Depth-Anything-V2', 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'),
    'MobileSAM': ('ChaoningZhang/MobileSAM', 'f706ad9c4eb7f219c00d9050e46328518ffb65d2'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', default='')
    args = parser.parse_args()
    env = dict(os.environ)
    if args.proxy:
        env.update(HTTP_PROXY=args.proxy, HTTPS_PROXY=args.proxy, ALL_PROXY=args.proxy)
    handlers = [urllib.request.ProxyHandler({'http': args.proxy, 'https': args.proxy})] if args.proxy else []
    opener = urllib.request.build_opener(*handlers)
    MODELS.mkdir(parents=True, exist_ok=True)

    def download(url, target):
        if target.exists() and target.stat().st_size > 1000:
            return
        temporary = target.with_suffix(target.suffix + '.part')
        print('Downloading', target.name, flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'icicle-pseudo-depth/1.0'})
        with opener.open(request, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        temporary.replace(target)

    revisions = {}
    for name, (repo, revision) in REPOS.items():
        target = MODELS / name
        if not target.exists():
            archive = MODELS / (name + '-' + revision[:8] + '.zip')
            download('https://codeload.github.com/' + repo + '/zip/' + revision, archive)
            with zipfile.ZipFile(archive) as z:
                # Upstream archives have a single top folder. Do not extract
                # archive paths outside our dedicated model directory.
                for item in z.infolist():
                    resolved = (MODELS / item.filename).resolve()
                    if MODELS.resolve() not in resolved.parents:
                        raise ValueError('Unsafe archive path')
                z.extractall(MODELS)
                extracted = MODELS / z.namelist()[0].split('/')[0]
            extracted.rename(target)
        revisions[name] = {'repository': 'https://github.com/' + repo, 'revision': revision}

    download('https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
             MODELS / 'depth_anything_v2_vits.pth')
    mobile_weight = MODELS / 'MobileSAM' / 'weights' / 'mobile_sam.pt'
    if not mobile_weight.exists():
        download('https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/' + REPOS['MobileSAM'][1]
                 + '/weights/mobile_sam.pt', mobile_weight)

    environment = ROOT / 'work' / 'pseudo_env'
    if not (environment / 'Scripts' / 'python.exe').exists():
        print('Creating isolated environment (shared CPU torch)', flush=True)
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(environment)
    python = environment / 'Scripts' / 'python.exe'
    packages = ['timm==0.9.16', 'huggingface-hub==0.26.2', 'safetensors==0.5.3']
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', *packages], env=env, check=True)
    checksums = {}
    for name, path in [('depth_anything', MODELS / 'depth_anything_v2_vits.pth'), ('mobile_sam', mobile_weight)]:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        checksums[name] = {'file': str(path.relative_to(ROOT)), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    lock = {'models': revisions, 'weights': checksums, 'optional_packages': packages,
            'python': str(python), 'base_python': sys.executable, 'device': 'cpu'}
    (MODELS / 'models.lock.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Ready:', python, flush=True)


if __name__ == '__main__':
    main()
