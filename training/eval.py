import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Mask2Former Swin-L on LaRS')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to checkpoint file (.pth).')
    parser.add_argument(
        '--config',
        default='configs/mask2former/mask2former_swin-l_lars_512x1024.py',
        help='Path to the evaluation config file.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/mask2former_swin-l_lars_512x1024/eval',
        help='Directory to save evaluation logs and artifacts.')
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
        'tools/test.py',
        args.config,
        args.checkpoint,
        '--work-dir',
        args.work_dir,
        '--launcher',
        args.launcher,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
