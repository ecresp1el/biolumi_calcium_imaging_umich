rule process_images:
    shell:
        "python BL_CalciumAnalysis/cli.py --mode process_all"