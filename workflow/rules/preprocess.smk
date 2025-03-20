rule process_images:
    input: "{folder}/raw/{filename}"
    output: "{folder}/processed/{filename}"
    shell:
        "python BL_CalciumAnalysis/cli.py --input {input} --output {folder}/processed --mode process_single"
