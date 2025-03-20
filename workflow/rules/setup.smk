rule setup_project:
    output: "config.json"
    shell:
        "python scripts/setup_project.py"
