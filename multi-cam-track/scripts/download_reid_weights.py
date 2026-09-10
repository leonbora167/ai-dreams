"""Download the recommended Torchreid person-ReID checkpoint.

This is an explicit operator-run step; importing this file does not download
anything. The checkpoint is the official Torchreid OSNet-AIN MSMT17 model.
"""
import argparse
from pathlib import Path


FILE_ID = '1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal'
DEFAULT_OUTPUT = Path('models/osnet_ain_x1_0_msmt17_256x128.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f'Already exists: {target}')
        return
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit('Missing gdown. Run: pip install gdown') from exc
    print(f'Downloading official Torchreid checkpoint to {target}')
    result = gdown.download(id=FILE_ID, output=str(target), quiet=False)
    if not result or not target.exists(): raise SystemExit('Checkpoint download failed.')
    print(f'ReID checkpoint ready: {target.resolve()}')


if __name__ == '__main__':
    main()
