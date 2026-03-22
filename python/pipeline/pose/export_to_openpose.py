#!/usr/bin/env python3
"""
Script to export pose data from HDF5 format to OpenPose JSON format

Usage:
    python export_to_openpose.py dataset.h5 output_directory [video_name]

Example:
    python export_to_openpose.py posedata.h5 ./openpose_output
    python export_to_openpose.py posedata.h5 ./openpose_output cam1
"""

import sys
import argparse
from pathlib import Path

# Import the MultiVideoPoseDataset class
from poseanalysis import MultiVideoPoseDataset


def main():
    parser = argparse.ArgumentParser(description='Export pose data to OpenPose JSON format')
    parser.add_argument('hdf5_file', help='Path to HDF5 pose dataset file')
    parser.add_argument('output_dir', help='Output directory for JSON files')
    parser.add_argument('video_name', nargs='?', help='Specific video to export (optional)')
    parser.add_argument('--summary', action='store_true', help='Show export summary before exporting')

    args = parser.parse_args()

    # Validate input file
    hdf5_path = Path(args.hdf5_file)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file '{args.hdf5_file}' not found")
        sys.exit(1)

    # Create dataset object
    dataset = MultiVideoPoseDataset(args.hdf5_file)

    try:
        # Show dataset info
        print("Dataset Information:")
        info = dataset.get_dataset_info()
        print(f"  File: {info['filepath']}")
        print(f"  Videos: {info['video_count']}")
        print(f"  Size: {info['file_size_mb']:.2f} MB")
        print(f"  Available videos: {info['video_names']}")
        print()

        # Determine which videos to export
        if args.video_name:
            if args.video_name not in info['video_names']:
                print(f"Error: Video '{args.video_name}' not found in dataset")
                print(f"Available videos: {info['video_names']}")
                sys.exit(1)
            videos_to_export = [args.video_name]
        else:
            videos_to_export = info['video_names']

        # Show export summary if requested
        if args.summary:
            print("Export Summary:")
            for video_name in videos_to_export:
                summary = dataset.get_export_summary(video_name)
                print(f"  {video_name}:")
                print(f"    Total frames: {summary['total_frames']}")
                print(f"    Persons: {summary['n_persons']}")
                print(f"    Keypoints per person: {summary['n_keypoints']}")
                for person_idx, frame_count in summary['person_frame_counts'].items():
                    print(f"    Person {person_idx}: {frame_count} valid frames")
            print()

        # Export to OpenPose JSON format
        print("Starting export...")
        for video_name in videos_to_export:
            dataset.export_video_to_openpose_json(video_name, args.output_dir)

        print(f"\nExport completed! JSON files saved to: {args.output_dir}")

        # Show output structure
        output_path = Path(args.output_dir)
        print("\nOutput structure:")
        for video_name in videos_to_export:
            video_dir = output_path / f"{video_name}_json"
            if video_dir.exists():
                json_files = list(video_dir.glob("*.json"))
                print(f"  {video_dir}: {len(json_files)} JSON files")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
