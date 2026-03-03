import os
import sys
from pathlib import Path


def enhanced_directory_structure(path, prefix="", max_depth=None, current_depth=0,
                                 exclude_dirs=None, exclude_extensions=None,
                                 show_size=False, show_hidden=False):
    """
    Enhanced version with more options.

    Args:
        path: Path to traverse
        prefix: Current line prefix
        max_depth: Maximum depth to traverse
        current_depth: Current depth
        exclude_dirs: List of directory names to exclude
        exclude_extensions: List of file extensions to exclude (e.g., ['.pyc', '.obj'])
        show_size: Show file sizes
        show_hidden: Show hidden files/directories
    """
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__', '.git', 'node_modules', 'build', 'dist']

    if exclude_extensions is None:
        exclude_extensions = ['.pyc', '.obj', '.exp', '.lib', '.o', '.d']

    if max_depth is not None and current_depth > max_depth:
        return

    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return

    # Filter items
    filtered_items = []
    for item in items:
        # Skip hidden files if not showing them
        if not show_hidden and item.startswith('.'):
            continue

        # Skip excluded directories
        if item in exclude_dirs:
            continue

        full_path = os.path.join(path, item)

        # Skip files with excluded extensions
        if os.path.isfile(full_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in exclude_extensions:
                continue

        filtered_items.append(item)

    for index, item in enumerate(filtered_items):
        full_path = os.path.join(path, item)
        is_last = (index == len(filtered_items) - 1)

        # Get file size if requested
        size_info = ""
        if show_size and os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            if size < 1024:
                size_info = f" ({size} B)"
            elif size < 1024 * 1024:
                size_info = f" ({size / 1024:.1f} KB)"
            else:
                size_info = f" ({size / (1024 * 1024):.1f} MB)"

        if os.path.isdir(full_path):
            if is_last:
                print(f"{prefix}└── 📁 {item}/")
                enhanced_directory_structure(full_path, prefix + "    ",
                                             max_depth, current_depth + 1,
                                             exclude_dirs, exclude_extensions,
                                             show_size, show_hidden)
            else:
                print(f"{prefix}├── 📁 {item}/")
                enhanced_directory_structure(full_path, prefix + "│   ",
                                             max_depth, current_depth + 1,
                                             exclude_dirs, exclude_extensions,
                                             show_size, show_hidden)
        else:
            if is_last:
                print(f"{prefix}└── 📄 {item}{size_info}")
            else:
                print(f"{prefix}├── 📄 {item}{size_info}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Display directory structure recursively')
    parser.add_argument('path', nargs='?', default='.', help='Path to analyze')
    parser.add_argument('--depth', '-d', type=int, help='Maximum depth to traverse')
    parser.add_argument('--output', '-o', help='Save output to file')
    parser.add_argument('--exclude-dirs', '-e', nargs='+',
                        default=['.venv', '__pycache__', '.git', 'node_modules', 'build_cmake'],
                        help='Directories to exclude')
    parser.add_argument('--exclude-ext', '-x', nargs='+',
                        default=['.pyc', '.obj', '.exp', '.lib', '.o', '.d'],
                        help='File extensions to exclude')
    parser.add_argument('--show-size', '-s', action='store_true',
                        help='Show file sizes')
    parser.add_argument('--show-hidden', action='store_true',
                        help='Show hidden files/directories')

    args = parser.parse_args()

    print(f"\n📁 Analyzing: {os.path.abspath(args.path)}")
    print(f"🚫 Excluding dirs: {', '.join(args.exclude_dirs)}")
    print(f"🚫 Excluding extensions: {', '.join(args.exclude_ext)}")
    if args.depth:
        print(f"📊 Max depth: {args.depth}")
    print()

    enhanced_directory_structure(args.path,
                                 max_depth=args.depth,
                                 exclude_dirs=args.exclude_dirs,
                                 exclude_extensions=args.exclude_ext,
                                 show_size=args.show_size,
                                 show_hidden=args.show_hidden)