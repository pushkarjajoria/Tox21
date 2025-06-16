import unittest
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from find_similar_molecules_tensor import tensorized_tanimoto_similarity


def smiles_to_morgan_fingerprint(smile, radius=2, n_bits=2048):
    """Convert SMILES string to Morgan fingerprint."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros(n_bits)  # Return a zero vector for invalid SMILES
    fp = mfpgen.GetFingerprint(mol)
    return fp


def rdkit_tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


class TestTensorizedTanimoto(unittest.TestCase):
    smiles_list = [
        "CCO",  # Ethanol
        "CCC",  # Propane
        "CCN",  # Ethylamine
        "CCOCC",  # Diethyl ether
        "CC(=O)O",  # Acetate
        "CCOCCO",  # Ethylene glycol diethyl ether
        "CC(C)O",  # Isopropanol
        "C1CCCCC1",  # Cyclohexane
        "C1=CC=CC=C1",  # Benzene
        "CC(C)C",  # Isobutane

        # Pharmaceutical compounds:
        "CC(C)Cc1ccc(cc1)O",  # Ibuprofen (Painkiller, NSAID)
        "COc1ccc(cc1)C(O)CN(C)C",  # Paracetamol (Painkiller, Fever reducer)
        "CCN1C(=O)c2c(ncn2C)n(c1=O)C",  # Caffeine (Stimulant)
        "CCOC(=O)c1c(cc(cc1O)O)C(=O)O",  # Aspirin (Painkiller, NSAID)
        "CN(C)C(=O)c1ccc(cc1)O",  # Acetaminophen (Analgesic)
        "COC(=O)c1c(nc(nc1O)N)N",  # Methotrexate (Chemotherapy, Autoimmune diseases)
        "CN(C)C(=O)c1c(cc(cc1O)O)O",  # Dopamine (Neurotransmitter, Parkinson's treatment)
        "O=C(O)c1ccccc1O",  # Salicylic Acid (Pain relief, Skincare)
        "CN1CCN(CC1)C(c2ccc(cc2)Cl)c3ncccc3",  # Sertraline (Antidepressant, SSRI)
        "CCN(CC)C(=O)c1c(cc(cc1O)O)O",  # Epinephrine (Adrenaline, Emergency medicine)
    ]

    def test_tanimoto_similarity(self):
        # Create test molecules

        # Generate fingerprints using RDKit
        rdkit_fps = [smiles_to_morgan_fingerprint(s) for s in self.smiles_list]
        rdkit_fp_array = np.array([list(fp) for fp in rdkit_fps])
        torch_fp_array = torch.tensor(rdkit_fp_array, dtype=torch.float32)

        # Compute pairwise similarity using RDKit
        rdkit_similarities = np.zeros((len(rdkit_fps), len(rdkit_fps)))
        for i in range(len(rdkit_fps)):
            for j in range(len(rdkit_fps)):
                rdkit_similarities[i, j] = rdkit_tanimoto_similarity(
                    rdkit_fps[i], rdkit_fps[j]
                )

        # Compute pairwise similarity using tensorized function
        tensorized_similarities = tensorized_tanimoto_similarity(torch_fp_array, torch_fp_array).cpu().numpy()

        # Check consistency
        np.testing.assert_allclose(
            tensorized_similarities, rdkit_similarities, rtol=1e-6, atol=1e-6
        )

    def test_tanimoto_similarity_with_feature_mask(self):
        # Generate fingerprints using RDKit
        rdkit_fps = [smiles_to_morgan_fingerprint(s) for s in self.smiles_list]
        rdkit_fp_array = np.array([list(fp) for fp in rdkit_fps])
        torch_fp_array = torch.tensor(rdkit_fp_array, dtype=torch.float32)

        # Create a feature mask where half the features are weighted higher
        feature_mask = np.random.rand(rdkit_fp_array.shape[1]).astype(np.float32)
        feature_mask /= feature_mask.sum()  # Normalize to sum to 1

        # Compute pairwise similarity using tensorized function with feature mask
        tensorized_similarities = tensorized_tanimoto_similarity(
            torch_fp_array, torch_fp_array, feature_mask
        ).cpu().numpy()

        # Ensure the output is valid (i.e., values are between 0 and 1)
        assert np.all((tensorized_similarities >= 0) & (tensorized_similarities <= 1))

        # Ensure diagonal elements are 1 (self-similarity)
        np.testing.assert_allclose(np.diag(tensorized_similarities), np.ones(len(self.smiles_list)), rtol=1e-6,
                                   atol=1e-6)


if __name__ == "__main__":
    unittest.main()
