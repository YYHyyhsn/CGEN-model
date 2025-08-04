from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Return train/val/test to the three DataLoaders.

    How to use:
        - If the directory where the dataset is located contains train.csv / val.csv / test.csv,
          The division will be made using the samples listed in these documents directly;
        - Otherwise, it will be automatically prorated from the dataset.id_prop_data (i.e. id_prop.csv).

    Parameter description:
            - dataset: CIFData instance (for reading root_dir)
            - train_ratio, val_ratio, test_ratio: Ratio when automatically divided
            - train_size, val_size, test_size: Specifying the number of samples (preferential over proportionality)
            - return_test: Whether to return test_loader
            - collate_fn: A splicing function for batching, which defaults to collate_pool
    """
    root_dir = dataset.root_dir

    # Check if there are three explicit partition files train.csv / val.csv / test.csv
    has_split_files = all(
        os.path.exists(os.path.join(root_dir, f"{split}.csv"))
        for split in ['train', 'val', 'test']
    )

    if has_split_files:
        print("[Info] Using predefined train/val/test split CSVs.")

        # If the user provides train.csv / val.csv / test.csv, the corresponding dataset is directly constructed
        train_dataset = CIFData(root_dir, split='train')
        val_dataset = CIFData(root_dir, split='val')
        test_dataset = CIFData(root_dir, split='test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)

    else:
        print("[Info] No train/val/test CSVs found. Automatically splitting dataset.")

        # Automatic Partitioning: Randomly divide train/val/test from the full dataset
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        # Try to read the explicitly specified number of samples first, otherwise use the scale
        if kwargs.get('train_size') is not None:
            train_size = kwargs['train_size']
        else:
            train_ratio = train_ratio or (1 - val_ratio - test_ratio)
            train_size = int(train_ratio * total_size)

        if kwargs.get('val_size') is not None:
            val_size = kwargs['val_size']
        else:
            val_size = int(val_ratio * total_size)

        if kwargs.get('test_size') is not None:
            test_size = kwargs['test_size']
        else:
            test_size = total_size - train_size - val_size

        assert train_size + val_size + test_size <= total_size, \
            "The total number of partitions exceeds the size of the dataset, please check the train/val/test parameter settings"

        # 按索引划分样本
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:train_size + val_size + test_size]

        # 使用 SubsetRandomSampler 构建 DataLoader
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader = DataLoader(dataset, batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=pin_memory)
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)

    # 根据 return_test 控制是否返回 test_loader
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, split=None):
        """
        split: one of ['train', 'val', 'test', None]
        """
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        self.split = split
        split_file = None
        if split is not None:
            split_file = os.path.join(root_dir, f'{split}.csv')
            assert os.path.exists(split_file), f'{split}.csv does not exist!'
        else:
            split_file = os.path.join(root_dir, 'id_prop.csv')
            assert os.path.exists(split_file), 'id_prop.csv does not exist!'

        with open(split_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # No shuffle if specific split is provided
        if split is None:
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        cif_path = os.path.join(self.root_dir, 'cif', cif_id + '.cif')
        assert os.path.exists(cif_path), f'{cif_path} does not exist!'
        crystal = Structure.from_file(cif_path)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f'{cif_id} not find enough neighbors. Consider increasing radius.')
                nbr_fea_idx.append([x[2] for x in nbr] + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append([x[1] for x in nbr] + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append([x[2] for x in nbr[:self.max_num_nbr]])
                nbr_fea.append([x[1] for x in nbr[:self.max_num_nbr]])
        nbr_fea = self.gdf.expand(np.array(nbr_fea))
        return (torch.Tensor(atom_fea),
                torch.Tensor(nbr_fea),
                torch.LongTensor(nbr_fea_idx)), \
               torch.Tensor([float(target)]), \
               cif_id

