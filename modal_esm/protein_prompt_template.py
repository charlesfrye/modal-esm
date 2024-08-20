import numpy as np
import torch
from typing import List, Optional, Set
from esm.sdk.api import ESMProtein
from biotite.structure import annotate_sse
from esm.utils.constants import esm3 as C

class ProteinPromptTemplate:
    def __init__(
        self, 
        sequence: str, 
        coordinates: torch.Tensor,
        secondary_structure: Optional[str] = None,
        fixed_seq_indices: Optional[List[int]] = None,
        fixed_struc_indices: Optional[List[int]] = None,
        
    ):
        self.seq_array = np.array(list(sequence))
        self.coord_array = coordinates
        self.length = len(self.seq_array)
        self.fixed_seq_indices: Set[int] = set(fixed_seq_indices or [])
        self.fixed_struc_indices: Set[int] = set(fixed_struc_indices or [])
        self.secondary_structure = secondary_structure
        self._update_p_mask()

    def _update_p_mask(self):
        self.p_mask = np.ones(self.length)
        fixed_indices = self.fixed_seq_indices.union(self.fixed_struc_indices)
        self.p_mask[list(fixed_indices)] = 0
        if self.p_mask.sum() > 0:
            self.p_mask /= self.p_mask.sum()
        else:
            # If all indices are fixed, set p_mask to all zeros
            self.p_mask[:] = 0

    @classmethod
    def from_esm_protein(
        cls, 
        esm_protein: ESMProtein,
        fixed_seq_indices: Optional[List[int]] = None,
        fixed_struc_indices: Optional[List[int]] = None
    ):
        return cls(
            sequence=esm_protein.sequence, 
            coordinates=esm_protein.coordinates, 
            secondary_structure=esm_protein.secondary_structure,
            fixed_seq_indices=fixed_seq_indices, 
            fixed_struc_indices=fixed_struc_indices)

    def apply_masking(self, mask_fraction: float = 0.05) -> 'ProteinPromptTemplate':
        mutable_indices = np.where(self.p_mask > 0)[0]
        num_to_mask = min(int(len(mutable_indices) * mask_fraction), len(mutable_indices))
        
        if num_to_mask == 0:
            return self  # Return a copy of self if no masking can be done
        
        mask_indices = np.random.choice(
            mutable_indices, 
            num_to_mask, 
            replace=False, 
            p=self.p_mask[mutable_indices] / self.p_mask[mutable_indices].sum()
        )
        
        new_seq = self.seq_array.copy()
        new_coord = self.coord_array.clone()
        
        new_seq[mask_indices] = '_'
        new_coord[mask_indices] = float('nan')
        
        return ProteinPromptTemplate(
            sequence=''.join(new_seq), 
            coordinates=new_coord, 
            secondary_structure=self.secondary_structure,
            fixed_seq_indices=self.fixed_seq_indices, 
            fixed_struc_indices=self.fixed_struc_indices
        )

    def mask_residue(self, index: int) -> 'ProteinPromptTemplate':
        if index in self.fixed_seq_indices or index in self.fixed_struc_indices:
            return self

        new_seq = self.seq_array.copy()
        new_coord = self.coord_array.clone()
        
        new_seq[index] = '_'
        new_coord[index] = float('nan')
        
        return ProteinPromptTemplate(
            ''.join(new_seq), 
            new_coord, 
            list(self.fixed_seq_indices), 
            list(self.fixed_struc_indices)
        )

    def get_masked_indices(self) -> List[int]:
        return np.where(self.seq_array == '_')[0].tolist()

    def to_esm_protein(self) -> ESMProtein:
        return ESMProtein(
            sequence=''.join(self.seq_array),
            coordinates=self.coord_array,
            secondary_structure=self.secondary_structure,
        )

    def __len__(self):
        return self.length
    