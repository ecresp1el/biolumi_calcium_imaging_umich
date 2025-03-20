rule finalize_results:
    input: "workflow/results/processed_images.txt"
    output: "workflow/results/final_report.txt"
    shell:
        "echo 'Processing completed successfully' > {output}"
