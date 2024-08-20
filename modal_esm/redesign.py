from dataclasses import dataclass
import os
from typing import Callable
import logging

from modal import App, Secret, gpu, Image, enter, method

from protein_prompt_template import ProteinPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAPROT_BASE_MODEL_PATH = "westlake-repl/SaProt_650M_AF2"
SAPROT_ADAPTERS = [
    {"name": "SaProtHub/Model-EYFP-650M", "key": "yfp_score"},
    {"name": "SaProtHub/Model-Fluorescence-650M", "key": "fluor_score"},
]
ESM3_PATH = "EvolutionaryScale/esm3-sm-open-v1"

def download_models():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    snapshot_download(repo_id=SAPROT_BASE_MODEL_PATH, repo_type="model")
    for adapter in SAPROT_ADAPTERS:
        snapshot_download(repo_id=adapter["name"], repo_type="model")
    snapshot_download(repo_id=ESM3_PATH, repo_type="model", token=os.environ["HUGGINGFACE_TOKEN"],)
    move_cache()


image = (
    Image.debian_slim()
    .pip_install("uv")
    .run_commands("uv pip install  --system --compile-bytecode torch transformers==4.28 huggingface_hub pandas tqdm peft esm")
    .run_function(download_models, secrets=[Secret.from_dotenv()])
    .env(dict(TOKENIZERS_PARALLELISM="false"))
    .run_commands(
            "apt-get update && apt-get install -y git build-essential wget zlib1g-dev libboost-all-dev",
            "wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh",
            "chmod +x cmake-3.26.4-linux-x86_64.sh",
            "./cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local",
            "rm cmake-3.26.4-linux-x86_64.sh",
            "git clone https://github.com/cmbi/dssp.git",
        )
    .run_commands(
        "cd dssp && ./autogen.sh && ./configure && make && make install",
        "ldconfig",  # Update shared library cache
        "ln -s /usr/local/bin/mkdssp /usr/local/bin/dssp",  # Create symlink for compatibility
        "cd .. && rm -rf dssp"
    )
)

with image.imports():
    import torch
    import numpy as np
    from typing import List, Tuple
    from esm.utils.structure.protein_chain import ProteinChain
    from esm.utils.structure.aligner import Aligner
    from esm.sdk.api import ESMProtein, GenerationConfig, ESM3InferenceClient
    from esm.tokenization import EsmSequenceTokenizer
    from esm.models.esm3 import ESM3
    from biotite.structure import annotate_sse
    from transformers import EsmTokenizer, EsmForSequenceClassification
    from peft import PeftConfig, PeftModelForSequenceClassification
    from huggingface_hub import snapshot_download
    import pandas as pd
    from tqdm import tqdm
    from esm.utils.constants import esm3 as C
    
@dataclass
class ProteinGenerationPrompt:
    prompt: "ESMProtein"
    sequence_config: "GenerationConfig"
    structure_config: "GenerationConfig"
    seed_sequence: str
    iteration_number: int
    
app = App(name='yfp-redesign', image=image)
    
@app.cls(
    gpu=gpu.A10G(), 
    container_idle_timeout=240, 
    image=image, 
    secrets=[Secret.from_dotenv()],
    concurrency_limit=20,
    )
class YfpRedesign:
    
    @enter()
    def enter(self):
        from esm.models.esm3 import ESM3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ESM3 model
        self.esm_model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        self.esm_tokenizer = EsmSequenceTokenizer()
        
        # SaProt model setup
        base_model = EsmForSequenceClassification.from_pretrained(SAPROT_BASE_MODEL_PATH, num_labels=1)
        adapter_path = snapshot_download(repo_id=SAPROT_ADAPTERS[0]["name"], repo_type="model")
        config = PeftConfig.from_pretrained(adapter_path)
        self.saprot_model = PeftModelForSequenceClassification(
            model=base_model,
            peft_config=config,
            adapter_name=SAPROT_ADAPTERS[0]['key'],
        )

        for adapter in SAPROT_ADAPTERS[1:]:
            adapter_path = snapshot_download(repo_id=adapter["name"], repo_type="model")
            config = PeftConfig.from_pretrained(adapter_path)
            self.saprot_model.add_adapter(adapter["key"], config)
        
        self.saprot_model.to(device)
        self.saprot_tokenizer = EsmTokenizer.from_pretrained(SAPROT_BASE_MODEL_PATH)
        
    @method()
    def run_redesign_iteration(
        self, 
        template: ProteinPromptTemplate,
        iteration: int,
        mask_fraction: float,
        fitness_function: Callable,
        variants_to_generate: int=50,
    ):
        
        # Generate "prompts" for esm3 generation - one for each variant
        generation_prompts = self.create_generation_prompts(
            template=template,
            variants_per_iteration=variants_to_generate, 
            mask_fraction=mask_fraction,
            iteration_number=iteration,
        )
        logger.info(f'created {len(generation_prompts)} generation prompts')

        # Generate variants using ESM3
        variants = self.generate_variants(generation_prompts)
        variants_df = self.evaluate_variants(variants, generation_prompts, template)
        
        # Add iteration number and append to all_variants_df
        variants_df['iteration'] = iteration
        variants_df['fitness'] = fitness_function(variants_df)
        return variants_df

    def create_generation_prompts(
        self, 
        sequence: str, 
        coords: torch.Tensor,
        variants_per_iteration: int, 
        sequence_fixed_indices: List[int], 
        structure_fixed_indices: List[int], 
        mask_fraction: float,
        iteration_number: int
    ) -> List[ProteinGenerationPrompt]:
        
        generation_prompts = []
        for _ in range(variants_per_iteration):
            # fix some indices in the sequence and structure, but allow others to be masked
            # fighting an urge to do this via dependency injection, since there are many ways one might want to mask
            prompt:ESMProtein = self.apply_masking(
                sequence=sequence, 
                coords=coords, 
                sequence_fixed_indices=sequence_fixed_indices, 
                structure_fixed_indices=structure_fixed_indices, 
                mask_fraction=mask_fraction
            )
            
            # inference configs for ESM
            sequence_config = GenerationConfig(
                track="sequence",
                num_steps=int(len(sequence) * mask_fraction), # this should be tunable to be fewer steps
            )
            structure_config = GenerationConfig(
                track="structure",
                num_steps=int(len(sequence) * mask_fraction),
                temperature=sequence_config.temperature
            )
            
            # package it all up
            gen_prompt = ProteinGenerationPrompt(
                prompt=prompt, 
                sequence_config=sequence_config, 
                structure_config=structure_config, 
                seed_sequence=sequence, 
                iteration_number=iteration_number
            )
            generation_prompts.append(gen_prompt)
        return generation_prompts

    def apply_masking(
        self, 
        sequence: str, 
        coords: torch.Tensor, 
        sequence_fixed_indices: List[int], 
        structure_fixed_indices: List[int],
        mask_fraction: float=0.05,
        method: str="random",
        
        ) -> "ESMProtein":
        
        
        num_residues = len(sequence)
        num_to_mask = int(num_residues * mask_fraction)
        all_indices = list(range(num_residues))

        if method == "random":
            to_mask = [i for i in all_indices if (i not in sequence_fixed_indices) or (i not in structure_fixed_indices)]
            mask_indices = np.random.choice(to_mask, num_to_mask, replace=False)
        elif method == "zeroshot":
            NotImplementedError("Zeroshot fitness masking not implemented")
        else:
            raise ValueError(f"Invalid method: {method}")
        
        masked_sequence = list(sequence)
        masked_coords = coords.clone()
        
        for idx in mask_indices:
            masked_sequence[idx] = '_'
            masked_coords[idx, :, :] = torch.full_like(masked_coords[idx, :, :], float('nan'))
        
        for idx in sequence_fixed_indices:
            masked_sequence[idx] = sequence[idx]
        
        for idx in structure_fixed_indices:
            masked_coords[idx, :, :] = coords[idx, :, :]
        
            return ESMProtein(sequence=''.join(masked_sequence), coordinates=masked_coords)
        
    def create_generation_prompts(
        self, 
        template: ProteinPromptTemplate,
        variants_per_iteration: int, 
        mask_fraction: float,
        iteration_number: int
    ) -> List[ProteinGenerationPrompt]:
        generation_prompts = []
        for _ in range(variants_per_iteration):
            prompt = template.apply_masking(mask_fraction=mask_fraction)
            sequence_config = GenerationConfig(
                track="sequence",
                num_steps=len(prompt.get_masked_indices()),
            )
            structure_config = GenerationConfig(
                track="structure",
                num_steps=len(prompt.get_masked_indices()),
                temperature=sequence_config.temperature
            )
            generation_prompts.append(ProteinGenerationPrompt(prompt.to_esm_protein(), sequence_config, structure_config, template.seq_array.tostring(), iteration_number))
        return generation_prompts

    def generate_variants(self, generation_prompts: List[ProteinGenerationPrompt]) -> List[ESMProtein]:
        # Generate variants using ESM
        variants = []
        for generation_prompt in tqdm(generation_prompts, total=len(generation_prompts)):
            # Generate sequence
            variant_sequence = self.esm_model.generate(generation_prompt.prompt, generation_prompt.sequence_config)
            
            # Generate structure
            variant = self.esm_model.generate(variant_sequence, generation_prompt.structure_config)
            variants.append(variant)
        
        return variants


    def evaluate_variants(
        self, 
        variants: List[ESMProtein], 
        generation_prompts: List[ProteinGenerationPrompt],
        template: ProteinPromptTemplate,
    ) -> pd.DataFrame:
        
        def calculate_sequence_identity(seq1: str, seq2: str) -> float:
            assert len(seq1) == len(seq2), "Sequences must be of equal length"
            identical = sum(a == b for a, b in zip(seq1, seq2))
            return identical / len(seq1)

        # TODO this is definitely broken lol
        def calculate_template_rmsd(variant: "ProteinChain", template: "ProteinChain", residues: List[int]) -> float:
            aligner = Aligner(variant, template)
            return aligner.rmsd

        def calculate_pseudo_perplexity(model, sequence: str) -> float:
            tokenizer = EsmSequenceTokenizer()
            tokens = torch.tensor([tokenizer.encode(sequence)]).to("cuda")
            with torch.no_grad():
                output = model(sequence_tokens=tokens)
            log_probs = torch.log_softmax(output.sequence_logits, dim=-1)
            token_log_probs = log_probs[0, torch.arange(len(sequence)), tokens[0, 1:-1]]
            return torch.exp(-token_log_probs.mean()).item()

        def calculate_n_gram_score(sequence: str, n: int = 3) -> float:
            from collections import Counter
            ngrams = [sequence[i:i+n] for i in range(len(sequence)-n+1)]
            counts = Counter(ngrams)
            return -sum(count * np.log(count/len(ngrams)) for count in counts.values()) / len(ngrams)

        def sequence_diff(ref: str, query: str) -> str:
            diff = []
            for i, (r, q) in enumerate(zip(ref, query)):
                if r != q:
                    diff.append(f"{r}{i+1}{q}")
            return '/'.join(diff)

        
        
        # Compute all evaluation metrics
        results = []
        for i, (variant, generation_prompt) in enumerate(zip(variants, generation_prompts)):
            seq_identity = calculate_sequence_identity(variant.sequence, ''.join(template.seq_array))
            chromophore_rmsd = calculate_template_rmsd(variant, template, [64, 65, 66])
            template_helix_rmsd = calculate_template_rmsd(variant, template, list(range(57, 71)))
            pseudo_perplexity = calculate_pseudo_perplexity(self.esm_model, variant.sequence)
            n_gram_score = calculate_n_gram_score(variant.sequence)
            
            # SaProt model evaluations
            sa_sequence = ''.join([aa + '#' for aa in variant.sequence]) + '#'
            inputs = self.saprot_tokenizer(sa_sequence, return_tensors="pt").to(self.saprot_model.device)
            
            scores = {}
            with torch.no_grad():
                for adapter in SAPROT_ADAPTERS:
                    self.saprot_model.set_adapter(adapter["key"])
                    scores[adapter["key"]] = self.saprot_model(**inputs).logits.item()
            
            # Compute diffs and mutation counts
            diff = sequence_diff(''.join(template.seq_array), variant.sequence)
            n_mutations = len(diff.split('/')) if diff else 0
            
            result = {
                "name": f'variant_{i}',
                "sequence": variant.sequence,
                "structure": variant.coordinates.tolist(),
                "seq_identity": seq_identity,
                "chromophore_rmsd": chromophore_rmsd,
                "template_helix_rmsd": template_helix_rmsd,
                "pseudo_perplexity": pseudo_perplexity,
                "n_gram_score": n_gram_score,
                "diff": diff,
                "n_mutations": n_mutations,
                "iteration": generation_prompt.iteration_number,
                "seed": generation_prompt.seed_sequence
            }
            
            result.update(scores)
            
            results.append(result)
        
        return pd.DataFrame(results)

def select_new_seeds(variants_df: "pd.DataFrame", n_select: int = 10) -> List[str]:

    # this is pretty underdeveloped
    # probably there should be a mixture of hard filtering + sampling and/or top K selection
    # should make more use of the individual metrics
    # would make sense to add something to encourage diversity among the selected seeds
    subset = variants_df.copy()
    
    # normalize fitness to be 0-1
    subset['fitness'] = (subset['fitness'] - subset['fitness'].min()) / (subset['fitness'].max() - subset['fitness'].min())
    fitness_weights = subset['fitness'] / subset['fitness'].sum()
    
    _n_select = min(n_select, len(subset))
    new_seeds = subset.sample(n=_n_select, weights=fitness_weights)['sequence'].tolist()
    return new_seeds
        

@app.function(timeout=1200)
def do_redesign(
    ref_sequence: str = 'MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGLQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSYQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK',
    sequence_fixed_indices = None,
    structure_fixed_indices= None,
    n_iterations = 20,
    n_processes = 20,
    n_generated_per_iter = 400,  # Changed from variants_per_iteration
    n_propagated_per_iter = 10,
    masking_function = None,
    fitness_function = None,
    add_approximate_ss_prompt:bool = True,
):
    # would be nice to write this in a less hardcoded way maybe
    if sequence_fixed_indices is None:
        sequence_fixed_indices = [1, 62, 65, 66, 67, 96, 222]
    if structure_fixed_indices is None:
        structure_fixed_indices = list(range(58, 72)) + [96, 222]
    if masking_function is None:
        masking_function = adjust_masking_rate
    if fitness_function is None:
        fitness_function = composite_fitness_func

    # Load YFP template
    pdb_id = "1YFP"  # Enhanced Yellow Fluorescent Protein
    chain_id = "A"
    ref_chain = ProteinChain.from_rcsb(pdb_id, chain_id)

    ref_pdb_start = ref_chain.residue_index[0]
    ref_pdb_end = ref_chain.residue_index[-1]
    end_padding = len(ref_sequence) - ref_pdb_end + 2

    coords = torch.tensor(ref_chain.atom37_positions)
    padded_coords = pad_coords(coords, ref_pdb_start, end_padding)
    
    template_esm_protein = ESMProtein(
        sequence=ref_sequence,
        coordinates=padded_coords,
    )
    
    
    
    if add_approximate_ss_prompt:
        protein_chain = template_esm_protein.to_protein_chain()
        secondary_structure = ''.join(protein_chain.dssp()).replace('X', C.MASK_STR_SHORT)
        template_esm_protein.secondary_structure = secondary_structure

    logger.info(template_esm_protein.secondary_structure)
    template = ProteinPromptTemplate.from_esm_protein(
        template_esm_protein,
        fixed_seq_indices=sequence_fixed_indices,
        fixed_struc_indices=structure_fixed_indices,
    )
    redesigner = YfpRedesign()
    
    seeds = [template] * n_processes
    iteration = 0

    # Calculate variants per process
    variants_per_process = n_generated_per_iter // n_processes
    remainder = n_generated_per_iter % n_processes

    redesign_iteration_configs = []
    for i, seed in enumerate(seeds):
        variants_for_this_process = variants_per_process + (1 if i < remainder else 0)
        redesign_iteration_configs.append((
            seed,
            iteration,
            masking_function(iteration),
            fitness_function,
            variants_for_this_process,
        ))

    all_results = []
    while iteration < n_iterations:
        for ii, result in enumerate(redesigner.run_redesign_iteration.starmap(redesign_iteration_configs)):
            result['chain'] = ii
            all_results.append(result)
        all_variants_df = pd.concat(all_results, ignore_index=True)
        # Select new seeds
        new_seed_sequences = select_new_seeds(all_variants_df, n_propagated_per_iter)
        seeds = [
            ProteinPromptTemplate(
                sequence=seq,
                coordinates=template.coord_array,
                secondary_structure=template.secondary_structure,
                fixed_seq_indices=template.fixed_seq_indices,
                fixed_struc_indices=template.fixed_struc_indices,
            )
            for seq in new_seed_sequences
        ]
        
        # Recalculate variants per process for the next iteration
        redesign_iteration_configs = []
        for i, seed in enumerate(seeds):
            variants_for_this_process = variants_per_process + (1 if i < remainder else 0)
            redesign_iteration_configs.append((
                seed,
                iteration,
                masking_function(iteration),
                fitness_function,
                variants_for_this_process,
            ))
        
        iteration += 1

    return all_variants_df
    

def pad_coords(coords, start_pad, end_pad):
    return torch.cat([
        torch.full((start_pad, 37, 3), float('nan')), 
        coords, 
        torch.full((end_pad, 37, 3), float('nan'))
        ], dim=0)

def adjust_masking_rate(iteration, initial_rate=0.2, decay_factor=0.9, floor_rate=0.01):
    return max(initial_rate * (decay_factor ** iteration), floor_rate)

def yfp_fitness_func(variants_df):
    return variants_df['yfp_score']

def composite_fitness_func(variants_df):
    return (variants_df['yfp_score'] + variants_df['fluor_score']*0.5) / 2
