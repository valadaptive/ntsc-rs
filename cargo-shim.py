# This script exists solely to copy build artifacts from Cargo's `target/debug` and `target/release` directories to the
# Meson root directory for building the After Effects plugin.
# TODO: replace this with a cargo xtask

import subprocess
import sys
import argparse
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument('--target-dir')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--release', action='store_true')
parser.add_argument('--target')

shim_args, cargo_args = parser.parse_known_args(sys.argv)

build_dir = shim_args.target_dir

if shim_args.release:
    cargo_args.append('--release')

if shim_args.target:
    cargo_args.append(f'--target={shim_args.target}')
    # If we specify a target triple, the artifacts will be placed under a directory whose name is that target triple
    build_dir = os.path.join(build_dir, shim_args.target)

build_dir = os.path.join(build_dir, 'release' if shim_args.release else 'debug')

cargo_cmd = ['cargo'] + cargo_args[1:] + ['--target-dir', shim_args.target_dir]
subprocess.run(cargo_cmd)
for file in os.listdir(build_dir):
    ext = os.path.splitext(file)[-1]
    if ext == '.a' or ext == '.dll' or ext == '.lib' or ext == '.so':
        shutil.copy2(os.path.join(build_dir, file), shim_args.out_dir)