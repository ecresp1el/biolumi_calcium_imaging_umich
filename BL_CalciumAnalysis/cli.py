"""
CLI for Calcium Imaging Analysis (Class-Based)

This script provides a structured way to process calcium imaging data from the command line.
It is based on the `ImageAnalysis` class and allows users to:
- Organize and analyze directories with imaging data.
- Apply specific image processing functions to sessions.
- Process single images or entire datasets.

Usage:
    - To process all images in a directory:
        python BL_CalciumAnalysis/cli.py --project_folder path/to/project --mode process_all

    - To process a single image:
        python BL_CalciumAnalysis/cli.py --input path/to/image.tif --output path/to/output/ --mode process_single

Arguments:
    --project_folder   : Path to the main project folder containing image datasets.
    --input            : Path to a specific image file for single processing.
    --output           : Path to the output directory for processed images.
    --mode             : Select between 'process_single' or 'process_all'.

Example:
    python BL_CalciumAnalysis/cli.py --project_folder data/raw --mode process_all
    python BL_CalciumAnalysis/cli.py --input data/raw/image.tif --output data/results --mode process_single

Author: Your Name
Date: 2025-03-20
"""

import argparse
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from BL_CalciumAnalysis.image_analysis_methods_umich import process_image

class ImageAnalysisCLI:
    """
    Class-based implementation for handling calcium imaging datasets.
    Supports organization, batch processing, and single-image processing.
    """

    def __init__(self, project_folder=None):
        self.project_folder = project_folder
        self.directory_df = self.initialize_directory_df() if project_folder else None

    def initialize_directory_df(self):
        """Initialize DataFrame containing directories within the project folder."""
        if not os.path.exists(self.project_folder):
            print(f"Error: Project folder '{self.project_folder}' does not exist.")
            return None
        
        directories = [d for d in os.listdir(self.project_folder) if os.path.isdir(os.path.join(self.project_folder, d))]
        directory_data = [{'directory_name': d, 'directory_path': os.path.join(self.project_folder, d)} for d in directories]
        return pd.DataFrame(directory_data, columns=['directory_name', 'directory_path'])

    def process_single_image(self, input_path, output_dir):
        """Process a single image file."""
        if not os.path.isfile(input_path):
            print(f"Error: Input file '{input_path}' does not exist.")
            return

        os.makedirs(output_dir, exist_ok=True)
        processed_image = process_image(input_path)
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        cv2.imwrite(output_path, processed_image)
        print(f"Processing complete -> Saved to {output_path}")

    def process_all_images(self, output_dir):
        """Process all images in the project folder directories."""
        if self.directory_df is None or self.directory_df.empty:
            print("Error: No directories found in the project folder.")
            return

        os.makedirs(output_dir, exist_ok=True)

        for _, row in self.directory_df.iterrows():
            folder_path = row['directory_path']
            print(f"Processing images in: {folder_path}")

            for filename in os.listdir(folder_path):
                if not (filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".jpg")):
                    print(f"Skipping non-image file: {filename}")
                    continue

                input_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_dir, filename)
                processed_image = process_image(input_path)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed {filename} -> Saved to {output_path}")

    def analyze_max_projection(self, tif_path):
        """
        Compute and save max intensity projection from a multi-frame TIF file.
        """
        with Image.open(tif_path) as img:
            sum_image = np.zeros((img.height, img.width), dtype=np.float32)

            for i in range(img.n_frames):
                img.seek(i)
                sum_image += np.array(img, dtype=np.float32)

            mean_image = sum_image / img.n_frames

        processed_dir = os.path.join(os.path.dirname(tif_path), 'processed_data', 'processed_image_analysis_output')
        os.makedirs(processed_dir, exist_ok=True)

        file_name = os.path.basename(tif_path)
        max_proj_image_path = os.path.join(processed_dir, file_name.replace('.tif', '_max_projection.tif'))
        Image.fromarray(mean_image).save(max_proj_image_path)

        print(f"Max projection saved: {max_proj_image_path}")
        return max_proj_image_path

def main():
    """
    Command-line interface for processing calcium imaging datasets.
    Allows users to process single images or entire datasets from a project folder.
    """

    parser = argparse.ArgumentParser(description="Process calcium imaging data from the command line.")

    parser.add_argument("--project_folder", type=str, help="Path to project folder (for batch processing).")
    parser.add_argument("--input", type=str, help="Path to a single image file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--mode", type=str, required=True, choices=['process_single', 'process_all'],
                        help="Mode: 'process_single' for one image, 'process_all' for batch processing.")

    args = parser.parse_args()

    if args.mode == "process_single":
        if not args.input:
            print("Error: --input is required for 'process_single' mode.")
            return
        ImageAnalysisCLI().process_single_image(args.input, args.output)

    elif args.mode == "process_all":
        if not args.project_folder:
            print("Error: --project_folder is required for 'process_all' mode.")
            return
        analysis = ImageAnalysisCLI(args.project_folder)
        analysis.process_all_images(args.output)

if __name__ == "__main__":
    main()