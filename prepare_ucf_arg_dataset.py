#!/usr/bin/env python3
"""
prepare_ucf_arg_dataset.py

Prepares a raw video dataset from the official UCF-ARG Ground Camera archive
for the AI-Powered Surveillance & Behavioral Intelligence System.

Extracts genuine videos for:
    - Walking
    - Jogging
    - Running

Organizes them into:
    dataset/
    └── raw/
        ├── Walking/
        ├── Jogging/
        └── Running/

This script does NOT perform feature extraction, pose estimation, or training.
Its sole responsibility is preparing the real video files for downstream processing
by dataset_generator.py.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Supported video file extensions
VALID_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".avi",
    ".mov",
    ".mpeg",
    ".mpg",
    ".mkv",
    ".m4v",
}

# Target behavior classes and their token mappings
TARGET_CLASS_TOKENS: Dict[str, str] = {
    "walking": "Walking",
    "walk": "Walking",
    "jogging": "Jogging",
    "jog": "Jogging",
    "running": "Running",
    "run": "Running",
}

# Non-target action classes in UCF-ARG to explicitly reject
EXCLUDED_ACTION_TOKENS: Set[str] = {
    "boxing",
    "box",
    "carrying",
    "carry",
    "clapping",
    "clap",
    "digging",
    "dig",
    "throwing",
    "throw",
    "waving",
    "wave",
    "trunk",
    "open",
    "close",
    "opening",
    "closing",
}


def tokenize_string(text: str) -> Set[str]:
    """Extract lowercase alphabetic words from a string or path segment."""
    return set(token for token in re.split(r"[^a-zA-Z]+", text.lower()) if token)


def classify_video(video_path: Path, base_extracted_dir: Path) -> Optional[str]:
    """
    Classifies a video file as 'Walking', 'Jogging', or 'Running' based on
    folder structure and filename conventions in the UCF-ARG dataset.

    Returns:
        Canonical class name ('Walking', 'Jogging', 'Running') or None if excluded/unmatched.
    """
    rel_path = video_path.relative_to(base_extracted_dir)

    # 1. Inspect directory hierarchy from deepest parent up to the extraction root
    for parent_part in reversed(rel_path.parts[:-1]):
        dir_tokens = tokenize_string(parent_part)

        # Check for direct match with target classes
        for token, canonical_name in TARGET_CLASS_TOKENS.items():
            if token in dir_tokens:
                return canonical_name

        # If the directory explicitly belongs to an excluded UCF-ARG action, discard immediately
        if dir_tokens & EXCLUDED_ACTION_TOKENS:
            return None

    # 2. If parent folders did not identify the class, inspect the filename stem
    stem_tokens = tokenize_string(video_path.stem)

    # Discard if filename belongs to an excluded UCF-ARG action (e.g., boxing_01, trunk_02)
    # unless a target keyword is also explicitly present
    if stem_tokens & EXCLUDED_ACTION_TOKENS and not (
        stem_tokens & set(TARGET_CLASS_TOKENS.keys())
    ):
        return None

    for token, canonical_name in TARGET_CLASS_TOKENS.items():
        if token in stem_tokens:
            return canonical_name

    return None


def get_unique_destination_path(target_dir: Path, original_filename: str) -> Path:
    """
    Resolves filename collisions by appending an incrementing counter suffix
    (e.g., 'video_1.avi', 'video_2.avi') without overwriting existing files.
    """
    destination = target_dir / original_filename
    if not destination.exists():
        return destination

    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix
    counter = 1

    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def prepare_dataset(
    archive_path_str: str,
    output_dir_str: str,
    verbose: bool = False,
) -> Tuple[Dict[str, int], Path]:
    """
    Validates, extracts, filters, and copies UCF-ARG Ground Camera videos.

    Args:
        archive_path_str: Path to the input ZIP archive.
        output_dir_str: Destination directory (e.g., 'dataset/raw').
        verbose: If True, prints individual file operations.

    Returns:
        Tuple of (class_counts dictionary, resolved output directory Path).
    """
    input_archive = Path(archive_path_str).resolve()
    output_base_dir = Path(output_dir_str).resolve()

    # Step 1: Validate input file existence
    if not input_archive.exists():
        print(f"Error: Input file does not exist: {input_archive}", file=sys.stderr)
        sys.exit(1)

    if not input_archive.is_file():
        print(f"Error: Input path is not a file: {input_archive}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Validate that file is a valid ZIP archive
    if not zipfile.is_zipfile(input_archive):
        print(
            f"Error: The provided input file is not a valid ZIP archive: {input_archive}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prepare target directories for the 3 target classes
    target_classes = ["Walking", "Jogging", "Running"]
    class_dirs: Dict[str, Path] = {}
    class_counts: Dict[str, int] = {cls: 0 for cls in target_classes}

    for cls in target_classes:
        class_path = output_base_dir / cls
        class_path.mkdir(parents=True, exist_ok=True)
        class_dirs[cls] = class_path

    # Step 3: Extract the archive into a secure temporary directory
    print(f"Extracting archive: {input_archive.name} ...")
    with tempfile.TemporaryDirectory(prefix="ucf_arg_ground_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            with zipfile.ZipFile(input_archive, "r") as archive:
                archive.extractall(temp_dir)
        except Exception as exc:
            print(f"Error: Failed to extract ZIP archive: {exc}", file=sys.stderr)
            sys.exit(1)

        print("Inspecting extracted files and identifying target behaviors...")

        # Step 4 & 15: Recursively inspect files with supported extensions
        discovered_videos: List[Path] = []
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in VALID_VIDEO_EXTENSIONS:
                discovered_videos.append(file_path)

        if not discovered_videos:
            print(
                f"Error: No supported video files found inside {input_archive.name}.\n"
                f"Supported extensions: {', '.join(sorted(VALID_VIDEO_EXTENSIONS))}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Step 5, 6, 8, 9, 10: Classify and copy target videos
        classified_videos: List[Tuple[Path, str]] = []
        for video in discovered_videos:
            label = classify_video(video, temp_dir)
            if label and label in target_classes:
                classified_videos.append((video, label))

        # Step 14: Error handling for no target classes found
        if not classified_videos:
            print(
                f"Error: None of the requested behavior classes ('Walking', 'Jogging', 'Running')\n"
                f"were found among the {len(discovered_videos)} video files in {input_archive.name}.\n"
                f"Please ensure this archive contains the genuine UCF-ARG Ground Camera dataset.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Copy files to designated class directories without recompression
        for src_path, label in classified_videos:
            dest_dir = class_dirs[label]
            dest_path = get_unique_destination_path(dest_dir, src_path.name)

            shutil.copy2(src_path, dest_path)
            class_counts[label] += 1

            if verbose:
                print(f"  [{label}] {src_path.name} -> {dest_path.name}")

    # Check for empty classes and display informative warning
    for cls in target_classes:
        if class_counts[cls] == 0:
            print(
                f"Warning: No videos found for class '{cls}' in the provided archive.",
                file=sys.stderr,
            )

    return class_counts, output_base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract and organize real UCF-ARG Ground Camera videos for Walking, Jogging, and Running."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the real UCF-ARG Ground Camera ZIP archive.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dataset/raw",
        help="Target output directory for raw organized videos (default: dataset/raw).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output showing each copied file.",
    )

    args = parser.parse_args()

    counts, out_dir = prepare_dataset(
        archive_path_str=args.input,
        output_dir_str=args.output,
        verbose=args.verbose,
    )

    total_videos = sum(counts.values())

    # Step 11 & 12: Print summary exactly as requested
    print("\nDataset preparation complete.\n")
    print(f"Walking: {counts['Walking']} videos")
    print(f"Jogging: {counts['Jogging']} videos")
    print(f"Running: {counts['Running']} videos")
    print(f"Total: {total_videos} videos\n")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
