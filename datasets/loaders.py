from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle as pkl
from einops import rearrange


class PDEDataset(Dataset):
    def __init__(self, path, num_env, n_data_per_env, num_var, model="moe", is_adapt=False):
        self.path = path
        self.num_env = num_env
        self.n_data_per_env = n_data_per_env
        self.num_var = num_var
        self.len = self.num_env * self.n_data_per_env
        with open(self.path, "rb") as f:
            self.data = pkl.load(f)

        self.model = model
        self.is_adapt = is_adapt

    def __getitem__(self, index):
        env_index = index // self.n_data_per_env
        data_index = index % self.n_data_per_env
        data = self.data[f"env_{env_index}"][data_index]
        state = torch.from_numpy(data["state"]).to(torch.float32)  # (Nx, Ny, C, T)
        context = torch.tensor(list(data["env_params"].values())).to(torch.float32)  # pde physical parameters
        time = torch.from_numpy(data["t_step"]).float()

        return {"state": state, "context": context, "time": time, "env_index": env_index, "sample_index": index}

    def __len__(self):
        return self.len


def get_pde_dataloader(cfg_data):
    dataset_train = PDEDataset(cfg_data["path_train"], cfg_data["num_env_train"], cfg_data["n_data_per_env_train"], cfg_data["num_var"])
    dataset_test_id = PDEDataset(cfg_data["path_test_id"], cfg_data["num_env_test"], cfg_data["n_data_per_env_test"], cfg_data["num_var"])
    dataset_test_ood = PDEDataset(cfg_data["path_test_ood"], cfg_data["num_env_test"], cfg_data["n_data_per_env_test"], cfg_data["num_var"])
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=cfg_data["batch_size_train"],
                                  num_workers=0,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
    dataloader_test_id = DataLoader(dataset=dataset_test_id,
                                    batch_size=cfg_data["batch_size_test"],
                                    num_workers=0,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False)
    dataloader_test_ood = DataLoader(dataset=dataset_test_ood,
                                     batch_size=cfg_data["batch_size_test"],
                                     num_workers=0,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False)
    return dataloader_train, dataloader_test_id, dataloader_test_ood
