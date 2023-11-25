print(
    f"""
PARAMS=(
)
sbatch launcher.sl lightning_main_pretraining.py ${{PARAMS[@]}}
"""
)
