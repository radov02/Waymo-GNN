"""Prepare VectorNet Data from Waymo TFRecord Files.

This script creates symbolic links or copies Waymo scenario TFRecord files
to the expected directory structure for VectorNet training.

VectorNetTFRecordDataset expects:
- data/scenario/training/*.tfrecord
- data/scenario/validation/*.tfrecord
- data/scenario/testing/*.tfrecord

Usage:
    # Create symbolic links (recommended, no disk space used)
    python src/vectornet/prepare_vectornet_data.py --source_dir /path/to/waymo/scenarios --link
    
    # Copy files (if symlinks not supported)
    python src/vectornet/prepare_vectornet_data.py --source_dir /path/to/waymo/scenarios --copy
    
    # Just check what would be done
    python src/vectornet/prepare_vectornet_data.py --source_dir /path/to/waymo/scenarios --dry-run
"""

import os
import sys
import glob
import shutil
import argparse
from pathlib import Path


def find_tfrecord_files(base_dir):
    """Find all TFRecord files in directory structure.
    
    Looks for patterns like:
    - training/*.tfrecord
    - validation/*.tfrecord  
    - testing/*.tfrecord
    """
    splits = {}
    
    # Try to find scenario subdirectory structure
    scenario_dir = os.path.join(base_dir, 'scenario')
    if os.path.exists(scenario_dir):
        base_dir = scenario_dir
    
    for split in ['training', 'validation', 'testing']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            files = sorted(glob.glob(os.path.join(split_dir, '*.tfrecord')))
            if files:
                splits[split] = files
                print(f"Found {len(files)} TFRecord files in {split_dir}")
        else:
            # Try without split subdirectory
            pattern = os.path.join(base_dir, f'*{split}*.tfrecord')
            files = sorted(glob.glob(pattern))
            if files:
                splits[split] = files
                print(f"Found {len(files)} {split} TFRecord files in {base_dir}")
    
    return splits


def setup_target_directories(target_base='data/scenario'):
    """Create target directory structure."""
    for split in ['training', 'validation', 'testing']:
        target_dir = os.path.join(target_base, split)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created/verified directory: {target_dir}")


def create_links(source_files, target_dir, mode='link', dry_run=False):
    """Create symbolic links or copies.
    
    Args:
        source_files: List of source file paths
        target_dir: Target directory
        mode: 'link', 'copy', or 'dry-run'
        dry_run: If True, just print what would be done
    """
    os.makedirs(target_dir, exist_ok=True)
    
    for source_file in source_files:
        filename = os.path.basename(source_file)
        target_file = os.path.join(target_dir, filename)
        
        # Check if target already exists
        if os.path.exists(target_file):
            if os.path.islink(target_file):
                # Remove existing symlink
                if not dry_run:
                    os.remove(target_file)
                print(f"  Removing existing link: {target_file}")
            elif os.path.isfile(target_file):
                # File already exists
                print(f"  File already exists: {target_file}")
                continue
        
        if dry_run:
            print(f"  Would {'link' if mode == 'link' else 'copy'}: {source_file} -> {target_file}")
        else:
            try:
                if mode == 'link':
                    # Create symbolic link
                    os.symlink(os.path.abspath(source_file), target_file)
                    print(f"  Linked: {target_file} -> {source_file}")
                elif mode == 'copy':
                    # Copy file
                    shutil.copy2(source_file, target_file)
                    print(f"  Copied: {source_file} -> {target_file}")
            except Exception as e:
                print(f"  ERROR: Failed to {mode} {source_file}: {e}")


def verify_setup(target_base='data/scenario'):
    """Verify that TFRecord files are accessible."""
    print("\n" + "="*60)
    print("Verifying setup...")
    print("="*60)
    
    total_files = 0
    for split in ['training', 'validation', 'testing']:
        target_dir = os.path.join(target_base, split)
        files = sorted(glob.glob(os.path.join(target_dir, '*.tfrecord')))
        print(f"{split:12s}: {len(files):4d} files")
        total_files += len(files)
    
    print(f"\nTotal: {total_files} TFRecord files")
    
    if total_files == 0:
        print("\n⚠️  WARNING: No TFRecord files found!")
        print("VectorNet training will fail. Please check source directory.")
        return False
    else:
        print("\n✅ Setup complete! Ready for VectorNet training.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare VectorNet data from Waymo TFRecord files'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Source directory containing TFRecord files (e.g., /path/to/waymo/scenario)'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='data/scenario',
        help='Target directory for VectorNet (default: data/scenario)'
    )
    parser.add_argument(
        '--link',
        action='store_true',
        help='Create symbolic links (recommended, saves disk space)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of linking'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.dry_run:
        mode = 'dry-run'
    elif args.copy:
        mode = 'copy'
    elif args.link:
        mode = 'link'
    else:
        print("Please specify --link, --copy, or --dry-run")
        return 1
    
    print("="*60)
    print("VectorNet Data Preparation")
    print("="*60)
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    print(f"Mode: {mode}")
    print("="*60)
    
    # Find source files
    print("\nSearching for TFRecord files...")
    splits = find_tfrecord_files(args.source_dir)
    
    if not splits:
        print(f"\n❌ ERROR: No TFRecord files found in {args.source_dir}")
        print("\nPlease check:")
        print("1. The path is correct")
        print("2. TFRecord files exist in training/validation/testing subdirectories")
        print("3. Files have .tfrecord extension")
        return 1
    
    # Setup target directories
    if not args.dry_run:
        print("\nSetting up target directories...")
        setup_target_directories(args.target_dir)
    
    # Create links/copies
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Processing files...")
    for split, files in splits.items():
        target_dir = os.path.join(args.target_dir, split)
        print(f"\n{split.upper()}:")
        create_links(files, target_dir, mode=mode, dry_run=args.dry_run)
    
    # Verify
    if not args.dry_run:
        verify_setup(args.target_dir)
    else:
        print("\n" + "="*60)
        print("DRY RUN complete - no files were modified")
        print("Remove --dry-run to perform actual setup")
        print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
