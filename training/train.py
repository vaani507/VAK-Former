import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask2Former Swin-L on LaRS')
    parser.add_argument(
        '--config',
        default='configs/mask2former/mask2former_swin-l_lars_512x1024.py',
        help='Path to the training config file.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/mask2former_swin-l_lars_512x1024',
        help='Directory to save logs and checkpoints.')
    parser.add_argument(
        '--launcher',
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help='Distributed launcher backend.')
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        'tools/train.py',
        args.config,
        '--work-dir',
        args.work_dir,
        '--launcher',
        args.launcher,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
