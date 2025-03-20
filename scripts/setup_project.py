import os
import json
import argparse

def create_project_structure(project_folder):
    """Creates a basic project folder and config.json"""
    
    # ✅ Ensure project folder exists
    os.makedirs(project_folder, exist_ok=True)

    # ✅ Define config.json path
    config_file = os.path.join(project_folder, "config.json")

    # ✅ Create a simple config.json
    config_data = {
        "project_root": project_folder,
        "status": "initialized"
    }

    # ✅ Save the config.json
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"✅ Project initialized at: {project_folder}")
    print(f"✅ Config file created at: {config_file}")

def main():
    """Parses arguments and initializes the project folder."""
    parser = argparse.ArgumentParser(description="Setup project directory.")
    parser.add_argument("--project_folder", type=str, required=True, help="Path to the project folder.")
    args = parser.parse_args()

    create_project_structure(args.project_folder)

if __name__ == "__main__":
    main()