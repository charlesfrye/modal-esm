import os
from modal import Image, App, method, enter, gpu

app = App("alphafold2-runner")

af2_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda11_pip]",
        "jaxlib",
        "numpy",
        "biopython",
        "pandas",
        "tensorflow",
        "dm-haiku",
        "ml-collections",
    )
    .apt_install("rsync", "wget", "git")
    .run_commands(
        "git clone https://github.com/nrbennet/dl_binder_design.git",
        "mkdir -p /data",
    )
    .run_commands(
        "cd /dl_binder_design && bash download_all_data.sh /data",
    )
)

@app.cls(image=af2_image, gpu=gpu.A100())
class AlphaFold2Runner:
    
    @enter()
    def enter(self):
        os.chdir("/dl_binder_design")

    @method()
    def predict(self, fasta_path: str, max_template_date: str = None, preset: str = "full_dbs"):
        args = ["--fasta_paths", fasta_path, "--preset", preset]
        if max_template_date:
            args.extend(["--max_template_date", max_template_date])
        
        command = f"python3 af2_initial_guess/predict.py {' '.join(args)}"
        print(f"Running command: {command}")
        os.system(command)
        
        os.system("zip -r output.zip output_dir")
        with open("output.zip", "rb") as f:
            return f.read()

@app.local_entrypoint()
def main(fasta_path: str=None, max_template_date: str = None, preset: str = "full_dbs"):
    runner = AlphaFold2Runner()
    result = runner.predict.remote(fasta_path, max_template_date, preset)
    
    with open("af2_output.zip", "wb") as f:
        f.write(result)
    print("Results saved to af2_output.zip")