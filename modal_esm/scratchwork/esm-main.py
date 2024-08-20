import io
from pathlib import Path

import safetensors

from modal import (
    App,
    Image,
    Mount,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
    Secret,
)
import numpy as np
import os

MODEL_DIR = "data"
MODEL_NAME="EvolutionaryScale/esm3-sm-open-v1"


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        # local_dir=model_dir,
        token=os.environ["HUGGINGFACE_TOKEN"],
        # ignore_patterns=["*.pt", "*.gguf"],  # Using safetensors
    )
    move_cache()



esm_image = (
    Image
    .debian_slim(python_version="3.10")
    # .poetry_install_from_file(
    #     "pyproject.toml",
    #     # pip_install_args=["--no-build-isolation"],
    # )
    .pip_install(
        "esm",
        "huggingface-hub",
    )
    .run_function(
        download_model_to_image,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME
            },
        secrets=[Secret.from_dotenv()]
    )
    .env(
        TOKENIZERS_PARALLELISM="false",
    )
    )



with esm_image.imports():
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, SamplingConfig, SamplingTrackConfig
    from esm.utils.structure.protein_chain import ProteinChain
    from esm.utils.types import FunctionAnnotation
    from esm.tokenization.function_tokenizer import (
        InterProQuantizedTokenizer as EsmFunctionTokenizer,
    )
    from esm.tokenization.sequence_tokenizer import (
        EsmSequenceTokenizer,
    )
    from esm.utils.constants.esm3 import (
        SEQUENCE_MASK_TOKEN,
)
    import torch
    import torch.nn.functional as F
    from esm.utils.constants import data_root
    
    
app = App(name='esm3-app', image=esm_image)
    
@app.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=esm_image, secrets=[Secret.from_dotenv()])
class ESM3App:
    
        
    @enter()
    def enter(self):
        self.model = ESM3.from_pretrained(
            "esm3_sm_open_v1",
            device="cuda",
            )
        
    def _fold(self, sequence, num_steps:int=50):
        protein = ESMProtein(sequence=sequence)
        return self.model.generate(protein, GenerationConfig(track='structure', num_steps=num_steps))
        
    @method()
    def fold(self, sequence: str, num_steps: int = 50) -> str:
        return self._fold(sequence, num_steps)
    
    
    def _mask_protein_regions(self, protein, seq_mask_regions=None, struct_mask_regions=None):
        """
        Mask specific regions of a protein's sequence and structure.
        
        Args:
        protein (ESMProtein): The protein to mask
        mask_regions (list of tuple): List of (start, end) tuples defining regions to mask
        
        Returns:
        ESMProtein: The masked protein
        """
        sequence = list(protein.sequence)
        coordinates = protein.coordinates.clone()
        seq_mask_regions = seq_mask_regions or []
        struct_mask_regions = struct_mask_regions or []
        
        for start, end in seq_mask_regions:
            sequence[start:end] = ["_"] * (end - start)
            
        for start, end in struct_mask_regions:
            coordinates[start:end] = torch.full((end - start, 37, 3), np.nan)
        
        return ESMProtein(sequence="".join(sequence), coordinates=coordinates)
    
    @method()
    def generate_with_masks(self, sequence: str, structure: str = None, 
                            seq_mask_regions: list[tuple[int, int]] = None,
                            struct_mask_regions: list[tuple[int, int]] = None,
                            num_steps: int = 8, temperature: float = 0.7) -> ESMProtein:
        """Generate a protein with multiple masked regions in sequence and/or structure."""
        
        protein = ESMProtein(sequence=sequence, coordinates=structure)
        
        if seq_mask_regions:
            protein = self.mask_protein_regions(protein, seq_mask_regions=seq_mask_regions)
        
        if structure and struct_mask_regions:
            protein = self.mask_protein_regions(protein, struct_mask_regions=struct_mask_regions)
        
        
        # Generate sequence if there are masked regions
        if '_' in protein.sequence:
            protein = self.model.generate(protein, GenerationConfig(
                track="sequence", num_steps=num_steps, temperature=temperature))
        
        # Generate structure if it's not provided or has masked regions
        if not protein.coordinates or (protein.coordinates and '_' in protein.coordinates):
            protein = self.model.generate(protein, GenerationConfig(
                track="structure", num_steps=num_steps))
        
        return protein
    

GFP = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'


@app.local_entrypoint()
def main(sequence=GFP):
    app = ESM3App()
    
    folded = app.generate_with_masks.remote(sequence=sequence)
    print(f"{folded}")
    