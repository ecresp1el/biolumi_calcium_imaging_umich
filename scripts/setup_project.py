import os
import json
import argparse

def create_project_structure(project_folder):
    """Prompt the user for details and create the project directory structure."""
    
    # ✅ Ensure the project folder exists but does NOT overwrite it
    os.makedirs(project_folder, exist_ok=True)

    # Check if `config.json` already exists
    config_file = os.path.join(project_folder, "config.json")
    if os.path.exists(config_file):
        print(f"⚠️  Warning: config.json already exists. Not overwriting.")
        return

    # Ask the user for the number of groups
    num_groups = int(input("🔹 Enter the number of groups: ").strip())

    config_data = {
        "project_root": project_folder,
        "groups": []
    }

    for i in range(1, num_groups + 1):
        group_name = input(f"🔹 Enter name for group {i}: ").strip()
        group_path = os.path.join(project_folder, group_name)
        os.makedirs(group_path, exist_ok=True)

        num_recordings = int(input(f"  🔸 Enter number of recordings for {group_name}: ").strip())

        group_data = {
            "group_name": group_name,
            "path": group_path,
            "recordings": []
        }

        for j in range(1, num_recordings + 1):
            recording_name = f"recording_{j:03d}"
            recording_path = os.path.join(group_path, recording_name)
            os.makedirs(recording_path, exist_ok=True)

            # ✅ Ensure these subdirectories exist but do NOT overwrite existing ones
            subdirs = ["raw", "metadata", "processed", "analysis", "figures"]
            for sub in subdirs:
                os.makedirs(os.path.join(recording_path, sub), exist_ok=True)

            group_data["recordings"].append({
                "recording_name": recording_name,
                "path": recording_path
            })

        config_data["groups"].append(group_data)

    # Save config.json inside the project folder
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"✅ Project initialized at: {project_folder}")
    print(f"✅ Config file created at: {config_file}")

def main():
    """Parse arguments and initialize the project folder."""
    parser = argparse.ArgumentParser(description="Setup project directory.")
    parser.add_argument("--project_folder", type=str, required=True, help="Path to the project folder.")
    args = parser.parse_args()

    create_project_structure(args.project_folder)

if __name__ == "__main__":
    main()