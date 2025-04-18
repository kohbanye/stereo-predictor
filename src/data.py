from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import numpy as np
import torch
from deepchem.feat import graph_features
from rdkit import Chem, rdBase
from rdkit.Chem import rdmolops
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

rdBase.DisableLog("rdApp.*")


class MoleculeDataset(Dataset):
    def __init__(self, sdf_file: str) -> None:
        self.sdf_file = sdf_file
        self.processed_file = sdf_file.replace(".sdf", ".pt")
        self.molecules = self.process_sdf()

    def process_sdf(self) -> list[Chem.Mol]:
        if Path(self.processed_file).exists():
            print(f"Loading processed SDF file from {self.processed_file}")
            with torch.serialization.safe_globals([Chem.Mol]):
                mols = cast(list[Chem.Mol], torch.load(self.processed_file))
                print(f"Loaded {len(mols)} molecules from {self.processed_file}")
                return mols

        molecules = []
        supplier = Chem.SDMolSupplier(self.sdf_file)
        for mol in tqdm(supplier, desc="Processing SDF file"):
            if mol is None:
                continue
            # keep only molecules with chiral centers
            chiral_centers = self.get_chiral_centers(mol)
            if len(chiral_centers) > 0:
                molecules.append(mol)

        print(
            f"Found {len(molecules)} molecules with chiral centers out of {len(supplier)} total molecules"
        )
        with torch.serialization.safe_globals([Chem.Mol]):
            torch.save(molecules, self.processed_file)
        return molecules

    def get_chiral_centers(self, mol: Chem.Mol) -> list[tuple[int, int]]:
        chiral_centers = []
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                # R = 1, S = 0
                chirality = (
                    1
                    if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                    else 0
                )
                chiral_centers.append((atom.GetIdx(), chirality))
        return chiral_centers

    def get_mol_graph(self, mol: Chem.Mol) -> Data:
        atom_features: list[np.ndarray] = []
        for atom in mol.GetAtoms():
            features = graph_features.atom_features(atom, use_chirality=False)
            atom_features.append(features)
        atom_features_tensor = torch.tensor(np.array(atom_features), dtype=torch.float)

        edge_features = []
        for bond in mol.GetBonds():
            features = graph_features.bond_features(bond, use_chirality=False)
            edge_features.append(features)
        adj_matrix: np.ndarray = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
        edge_features_tensor = torch.tensor(np.array(edge_features), dtype=torch.float)

        chiral_centers = self.get_chiral_centers(mol)

        # Create node-level labels with masking
        num_atoms = mol.GetNumAtoms()
        chiral_labels = torch.full(
            (num_atoms,), -100, dtype=torch.long
        )  # Default to ignore_index (-100)

        for idx, chirality in chiral_centers:
            chiral_labels[idx] = chirality

        chiral_indices = torch.tensor(
            [idx for idx, _ in chiral_centers], dtype=torch.long
        )

        return Data(
            x=atom_features_tensor,
            edge_index=edge_index,
            edge_attr=edge_features_tensor,
            y=chiral_labels,
            chiral_indices=chiral_indices,
        )

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Data:
        mol = self.molecules[idx]
        return self.get_mol_graph(mol)


class ChiralityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sdf_file: str,
        batch_size: int = 32,
        test_size: float = 0.1,
        val_size: float = 0.1,
        num_workers: int = 32,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.sdf_file = sdf_file
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.random_state = random_state

        self._dataset: MoleculeDataset | None = None
        self._train_indices: list[int] = []
        self._val_indices: list[int] = []
        self._test_indices: list[int] = []

    @property
    def dataset(self) -> MoleculeDataset:
        if self._dataset is None:
            self._dataset = MoleculeDataset(self.sdf_file)
        return self._dataset

    @property
    def num_node_features(self) -> int:
        return int(self.dataset[0].num_node_features)

    def prepare_data(self) -> None:
        # Initialize dataset
        _ = self.dataset

    def setup(self, stage: str | None = None) -> None:
        if not self._train_indices and (stage == "fit" or stage is None):
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))

            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            val_size_adjusted = self.val_size / (1 - self.test_size)
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size_adjusted,
                random_state=self.random_state,
            )

            self._train_indices = train_indices
            self._val_indices = val_indices
            self._test_indices = test_indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self._train_indices),
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self._val_indices),
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self._test_indices),
            num_workers=self.num_workers,
        )
