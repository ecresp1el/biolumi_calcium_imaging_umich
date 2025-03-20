import os
import json

# 🛠️ Helper function to get user input with a default option
def get_user_input(prompt, default=None):
    """Prompt the user for input with an optional default value."""
    response = input(f"{prompt} [{default}]: ") or default
    return response

# 🏗️ Function to create a structured project directory
def create_directory_structure(project_root, group_info):
    """Creates a standardized directory structure for calcium imaging projects."""
    
    # Ensure the project root directory exists
    os.makedirs(project_root, exist_ok=True)

    # Store project configuration in a dictionary
    project_config = {
        "project_root": project_root,
        "groups": []
    }

    # 🔹 Loop through each group to create folders
    for group_name, recordings_per_group in group_info.items():
        group_path = os.path.join(project_root, group_name)
        os.makedirs(group_path, exist_ok=True)

        group_data = {"group_name": group_name, "path": group_path, "recordings": []}

        # 🔹 Loop through each recording in the group
        for r in range(1, recordings_per_group + 1):
            recording_name = f"recording_{r:03d}"
            recording_path = os.path.join(group_path, recording_name)
            os.makedirs(recording_path, exist_ok=True)

            # Create subdirectories within the recording folder
            subdirs = ["raw", "metadata", "processed", "analysis", "figures"]
            for sub in subdirs:
                os.makedirs(os.path.join(recording_path, sub), exist_ok=True)

            # Add a README file to each recording folder
            with open(os.path.join(recording_path, "README.md"), "w") as readme:
                readme.write(f"# {recording_name}\n\nThis folder contains data for {recording_name}.")

            # Store recording details in the config
            group_data["recordings"].append({"recording_name": recording_name, "path": recording_path})

        # Add a README file to each group folder
        with open(os.path.join(group_path, "README.md"), "w") as readme:
            readme.write(f"# {group_name}\n\nThis folder contains multiple recording sessions.")

        # Store group details in the config
        project_config["groups"].append(group_data)

    # Create a README for the entire project
    with open(os.path.join(project_root, "README.md"), "w") as readme:
        readme.write(f"# {os.path.basename(project_root)}\n\nThis is a structured directory for calcium imaging data.")

    # Save the project structure configuration as a JSON file
    config_path = os.path.join(project_root, "config.json")
    with open(config_path, "w") as json_file:
        json.dump(project_config, json_file, indent=4)

    print(f"✅ Project structure created at: {project_root}")
    print(f"🔧 Configuration saved in: {config_path}")

# 🎯 Main function to prompt user input and create the directory
def main():
    """Main function to guide users through project setup."""
    print("🔹 Welcome to the Project Setup Script 🔹")

    # Get user input for project setup
    project_root = get_user_input("Enter project directory name", "biolumi_project")
    num_groups = int(get_user_input("How many groups?", "2"))

    group_info = {}
    for i in range(1, num_groups + 1):
        group_name = get_user_input(f"Enter name for group {i}", f"group_{i:03d}")
        recordings_per_group = int(get_user_input(f"How many recordings for {group_name}?", "2"))
        group_info[group_name] = recordings_per_group

    # Call function to create the structured directory tree
    create_directory_structure(project_root, group_info)

# 🚀 Run the script when executed
if __name__ == "__main__":
    main()