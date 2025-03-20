
#Biolumi Calcium Imaging Analysis

📌 Overview

This project provides a structured approach for analyzing calcium imaging data. It ensures reproducibility, organization, and automation by setting up a standardized directory structure and enabling image processing via the command line.

🛠️ Installation & Setup

1️⃣ Set Up the Conda Environment

Ensure you have Miniconda or Anaconda installed.

Run the following command to create and activate the environment:

2️⃣ Set Up the Project Directory Structure

Run the setup script to create a standardized directory tree for storing image data:

You'll be prompted to enter:

Project directory name (default: biolumi_project)

Number of groups (default: 2)

Number of recordings per group (default: 2)

This will create a structure like:

3️⃣ Process Images Using CLI

After setting up the structure, you can process images.

🔹 Process a Single Image

🔹 Process All Images in a Directory

4️⃣ Running Tests

To verify that everything works correctly, run:

🚀 Future Enhancements

Automate metadata collection.

Add parallel processing for large datasets.

Enhance logging and error handling.

This project is designed to be scalable, efficient, and user-friendly.Feel free to contribute or suggest improvements! 🚀