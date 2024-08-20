import modal
import zipfile
import tempfile
import shutil
from pathlib import Path
from esm.sdk.api import ESMProtein
from typing import Dict, Optional

def get_pdb_files(directory: Path) -> Dict[str, Path]:
    """
    Get all PDB files in the directory, sorted by relaxation status and rank.
    """
    pdb_files = sorted(directory.glob("*.pdb"))
    return {pdb.stem: pdb for pdb in pdb_files}

def parse_pdb_filename(filename: str) -> Dict[str, str]:
    """
    Parse the PDB filename to extract relevant information.
    """
    parts = filename.split('_')
    return {
        "relaxation": "relaxed" if "relaxed" in filename else "unrelaxed",
        "rank": next((part for part in parts if part.startswith("rank")), "rank999"),
        "model": next((part for part in parts if part.startswith("model")), "model_1"),
    }

def get_best_pdb(pdb_files: Dict[str, Path]) -> Optional[Path]:
    """
    Get the best PDB file based on relaxation status and rank.
    """
    if not pdb_files:
        return None
    
    parsed_files = [(parse_pdb_filename(name), path) for name, path in pdb_files.items()]
    
    # Sort by relaxation (relaxed first), then by rank (lower first)
    sorted_files = sorted(parsed_files, key=lambda x: (x[0]["relaxation"] != "relaxed", x[0]["rank"]))
    
    return sorted_files[0][1] if sorted_files else None

def sequence_to_esmproteins(sequence: str, keep_results: bool = False, output_dir: Optional[Path] = None) -> Dict[str, ESMProtein]:
    """
    Convert a protein sequence to ESMProtein objects using AlphaFold.
    
    :param sequence: str, the protein sequence
    :param keep_results: bool, whether to keep the results files (default: False)
    :param output_dir: Path, directory to save results if keep_results is True (default: None)
    :return: Dict[str, ESMProtein], dictionary of ESMProtein objects keyed by PDB filename
    """
    # Prepare FASTA input
    fasta = f"> Sequence\n{sequence}"
    
    # Run AlphaFold
    af = modal.Function.lookup("minimalaf", "fold")
    result = af.remote(fasta=fasta, models=[1], num_recycles=1)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "results.zip"
        
        # Save and extract results
        zip_path.write_bytes(result)
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)
        
        # Get all PDB files
        pdb_files = get_pdb_files(temp_path)
        
        if not pdb_files:
            raise FileNotFoundError("No .pdb files found in the results.")
        
        # Create ESMProtein objects
        esm_proteins = {name: ESMProtein.from_pdb(str(path)) for name, path in pdb_files.items()}
        
        # Keep results if requested
        if keep_results:
            if output_dir is None:
                output_dir = Path("af_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            for file in temp_path.iterdir():
                shutil.copy2(file, output_dir)
            print(f"Results saved in: {output_dir}")
        
    return esm_proteins

def fold(sequence: str, keep_results: bool = False, output_dir: Optional[Path] = None) -> ESMProtein:
    """
    Convert a protein sequence to the best ESMProtein object using AlphaFold.
    
    :param sequence: str, the protein sequence
    :param keep_results: bool, whether to keep the results files (default: False)
    :param output_dir: Path, directory to save results if keep_results is True (default: None)
    :return: ESMProtein, the best ESMProtein object
    """
    esm_proteins = sequence_to_esmproteins(sequence, keep_results, output_dir)
    best_pdb = get_best_pdb({name: Path(name) for name in esm_proteins.keys()})
    
    if best_pdb is None:
        raise ValueError("No valid PDB file found in the results.")
    
    return esm_proteins[best_pdb.stem]