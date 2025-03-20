"""
CLI for Calcium Imaging Analysis (Class-Based)

This script processes calcium imaging data from the command line.

🔹 Updates:
- Automatically detects `project_folder` from `config.json`
- Only processes images inside `recording_XXX/raw/` folders
- Saves results inside each `recording_XXX/processed/` folder
"""

import argparse
import os
import json
import cv2
import numpy as np
from PIL import Image
from BL_CalciumAnalysis.image_analysis_methods_umich import process_image

class ImageAnalysisCLI:
    """Class-based implementation for handling calcium imaging datasets."""

    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.config = self.load_config()

    def load_config(self):
        """Load project configuration from config.json."""
        config_path = os.path.join(self.project_folder, "config.json")
        if not os.path.exists(config_path):
            print(f"❌ Error: Configuration file '{config_path}' not found.")
            return None
        
        with open(config_path, "r") as json_file:
            return json.load(json_file)

    def process_single_image(self, input_path, output_dir):
        """Process a single image file."""
        if not os.path.exists(input_path):
            print(f"❌ Error: Input file '{input_path}' does not exist.")
            return

        os.makedirs(output_dir, exist_ok=True)
        processed_image = process_image(input_path)
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        cv2.imwrite(output_path, processed_image)
        print(f"✅ Processed {input_path} -> Saved to {output_path}")

    def process_all_images(self):
        """Process all images in the project."""
        if not self.config:
            return

        for group in self.config["groups"]:
            print(f"📂 Processing group: {group['group_name']}")

            for recording in group["recordings"]:
                raw_folder = os.path.join(recording["path"], "raw")
                processed_folder = os.path.join(recording["path"], "processed")

                os.makedirs(processed_folder, exist_ok=True)

                print(f"🔹 Processing recording: {recording['recording_name']}")

                for filename in os.listdir(raw_folder):
                    if filename.endswith(".tif") or filename.endswith(".tiff"):
                        input_path = os.path.join(raw_folder, filename)
                        output_path = os.path.join(processed_folder, filename)
                        processed_image = process_image(input_path)
                        cv2.imwrite(output_path, processed_image)
                        print(f"✅ Processed {filename} -> {output_path}")

def main():
    """Command-line interface for processing calcium imaging datasets."""
    parser = argparse.ArgumentParser(description="Process calcium imaging data from the command line.")
    parser.add_argument("--project_folder", type=str, help="Path to project folder (uses config.json).")
    parser.add_argument("--input", type=str, help="Path to a single image file.")
    parser.add_argument("--output", type=str, help="Path to output directory (for single image processing).")
    parser.add_argument("--mode", type=str, required=True, choices=['process_single', 'process_all'],
                        help="Mode: 'process_single' for one image, 'process_all' for batch processing.")

    args = parser.parse_args()

    if args.mode == "process_single":
        if not args.input or not args.output:
            print("❌ Error: --input and --output are required for 'process_single' mode.")
            return
        ImageAnalysisCLI(args.project_folder).process_single_image(args.input, args.output)

    elif args.mode == "process_all":
        if not args.project_folder:
            print("❌ Error: --project_folder is required for 'process_all' mode.")
            return
        analysis = ImageAnalysisCLI(args.project_folder)
        analysis.process_all_images()

if __name__ == "__main__":
    main()