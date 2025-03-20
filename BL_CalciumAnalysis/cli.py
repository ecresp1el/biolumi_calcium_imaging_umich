"""
CLI for Calcium Imaging Analysis

This script allows users to process calcium imaging data from the command line.
It can handle:
- Single image files
- Entire directories containing multiple images

Usage:
    - To process a single image:
        python BL_CalciumAnalysis/cli.py --input path/to/image.tif --output path/to/output/

    - To process all images in a directory:
        python BL_CalciumAnalysis/cli.py --input path/to/folder --output path/to/output/

Arguments:
    --input   : Path to an input image file or a folder containing images.
    --output  : Path to the output directory where processed images will be saved.

Example:
    python BL_CalciumAnalysis/cli.py --input data/raw --output data/results

Requirements:
    - OpenCV (`cv2`) for image processing
    - NumPy (`numpy`)
    - argparse (built into Python)
    - BL_CalciumAnalysis.image_analysis_methods_umich (processing functions)

Author: Your Name
Date: 2025-03-20
"""

import argparse
import os
import cv2
from BL_CalciumAnalysis.image_analysis_methods_umich import process_image

def main():
    """
    Main function that processes images from the command line.

    - Parses command-line arguments using argparse.
    - Checks if the input is a single image or a directory.
    - Processes each image using `process_image()` from `image_analysis_methods_umich.py`.
    - Saves processed images in the specified output directory.
    """

    # Step 1: Define the argument parser
    parser = argparse.ArgumentParser(
        description="Process calcium imaging data from the command line."
    )

    # Step 2: Add required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file or directory containing images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory where processed images will be saved.",
    )

    # Step 3: Parse the arguments
    args = parser.parse_args()

    # Step 4: Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Step 5: Process images
    if os.path.isdir(args.input):
        print(f"Processing all images in directory: {args.input}")

        for filename in os.listdir(args.input):
            input_path = os.path.join(args.input, filename)

            # Skip non-image files
            if not (filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".jpg")):
                print(f"Skipping non-image file: {filename}")
                continue

            output_path = os.path.join(args.output, filename)
            processed_image = process_image(input_path)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed {filename} -> Saved to {output_path}")

    else:
        print(f"Processing single image: {args.input}")
        
        # Check if input file exists
        if not os.path.isfile(args.input):
            print(f"Error: Input file '{args.input}' does not exist.")
            return

        processed_image = process_image(args.input)
        output_path = os.path.join(args.output, os.path.basename(args.input))
        cv2.imwrite(output_path, processed_image)
        print(f"Processing complete -> Saved to {output_path}")

if __name__ == "__main__":
    main()