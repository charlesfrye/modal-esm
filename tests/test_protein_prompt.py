import pytest
import numpy as np
import torch
from modal_esm.protein_prompt_template import ProteinPromptTemplate
from esm.sdk.api import ESMProtein

def test_initialization():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    coords = torch.randn(20, 37, 3)
    template = ProteinPromptTemplate(seq, coords, fixed_seq_indices=[0, 1], fixed_struc_indices=[2, 3])
    
    assert len(template) == 20
    assert np.array_equal(template.seq_array, np.array(list(seq)))
    assert torch.equal(template.coord_array, coords)
    assert template.fixed_seq_indices == {0, 1}
    assert template.fixed_struc_indices == {2, 3}
    assert np.allclose(template.p_mask[4:], 1/16)
    assert np.all(template.p_mask[:4] == 0)

def test_from_esm_protein():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    coords = torch.randn(20, 37, 3)
    esm_protein = ESMProtein(sequence=seq, coordinates=coords)
    template = ProteinPromptTemplate.from_esm_protein(
        esm_protein, 
        fixed_seq_indices=[0, 1], 
        fixed_struc_indices=[2, 3]
    )
    
    assert len(template) == 20
    assert np.array_equal(template.seq_array, np.array(list(seq)))
    assert torch.equal(template.coord_array, coords)
    assert template.fixed_seq_indices == {0, 1}
    assert template.fixed_struc_indices == {2, 3}
    assert np.allclose(template.p_mask[4:], 1/16)
    assert np.all(template.p_mask[:4] == 0)

def test_apply_masking():
    seq = "ACDEFG"
    coords = torch.randn(6, 37, 3)
    template = ProteinPromptTemplate(seq, coords, fixed_seq_indices=[0], fixed_struc_indices=[1])
    
    masked_template = template.apply_masking(mask_fraction=0.5)
    
    assert 1 <= len(masked_template.get_masked_indices()) <= 2
    assert masked_template.seq_array[0] != '_'
    assert masked_template.seq_array[1] != '_'
    assert not torch.isnan(masked_template.coord_array[0]).any()
    assert not torch.isnan(masked_template.coord_array[1]).any()

def test_mask_residue():
    seq = "ACDEFG"
    coords = torch.randn(6, 37, 3)
    template = ProteinPromptTemplate(seq, coords, fixed_seq_indices=[0], fixed_struc_indices=[1])
    
    masked_template = template.mask_residue(2)
    
    assert masked_template.seq_array[2] == '_'
    assert torch.isnan(masked_template.coord_array[2]).all()
    assert np.array_equal(masked_template.seq_array[:2], template.seq_array[:2])
    assert np.array_equal(masked_template.seq_array[3:], template.seq_array[3:])
    
    # Attempt to mask fixed residue
    masked_template = template.mask_residue(0)
    assert masked_template.seq_array[0] != '_'
    assert not torch.isnan(masked_template.coord_array[0]).any()

def test_get_masked_indices():
    template = ProteinPromptTemplate("AC_EF_", torch.randn(6, 37, 3))
    
    masked_indices = template.get_masked_indices()
    
    assert masked_indices == [2, 5]

def test_to_esm_protein():
    seq = "ACDEFG"
    coords = torch.randn(6, 37, 3)
    template = ProteinPromptTemplate(seq, coords)
    
    esm_protein = template.to_esm_protein()
    
    assert isinstance(esm_protein, ESMProtein)
    assert esm_protein.sequence == seq
    assert torch.equal(esm_protein.coordinates, coords)

def test_fixed_indices_respected():
    seq = "ACDEFG"
    coords = torch.randn(6, 37, 3)
    template = ProteinPromptTemplate(seq, coords, fixed_seq_indices=[0, 1], fixed_struc_indices=[2, 3])
    
    for _ in range(100):  # Run multiple times to account for randomness
        masked_template = template.apply_masking(mask_fraction=0.5)
        assert masked_template.seq_array[0] != '_'
        assert masked_template.seq_array[1] != '_'
        assert not torch.isnan(masked_template.coord_array[2]).any()
        assert not torch.isnan(masked_template.coord_array[3]).any()

def test_p_mask_update():
    seq = "ACDEFG"
    coords = torch.randn(6, 37, 3)
    template = ProteinPromptTemplate(seq, coords, fixed_seq_indices=[0, 1], fixed_struc_indices=[2, 3])
    
    assert np.allclose(template.p_mask[4:], 0.5)
    assert np.all(template.p_mask[:4] == 0)

    # Update fixed indices
    template.fixed_seq_indices.add(4)
    template._update_p_mask()

    assert np.allclose(template.p_mask[5], 1)
    assert np.all(template.p_mask[:5] == 0)

def test_all_fixed_indices():
    seq = "ACDEFG"
    coords = torch.randn(6, 37, 3)
    template = ProteinPromptTemplate(seq, coords, fixed_seq_indices=list(range(6)))
    
    assert np.all(template.p_mask == 0)
    
    masked_template = template.apply_masking(mask_fraction=0.5)
    assert np.array_equal(masked_template.seq_array, template.seq_array)
    assert torch.equal(masked_template.coord_array, template.coord_array)