"""
Script for data handling and processing for the MagNet challenge.

This module defines the `MagNetDataModule` class, which is responsible for loading,
preprocessing, and serving data to the MagNet model. It handles:
- Loading B-field, Frequency, and Temperature data.
- Interpolating B-field data to a fixed sample number.
- performing Log-transformation and normalization on inputs.
- Splitting data into training, validation, and test sets.
"""
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import lightning.pytorch as pl
import json

class EmptyDataset(Dataset):
    """
    A dummy dataset that returns nothing. Used for validation/training steps when
    running in inference mode to avoid errors with empty dataloaders.
    """
    def __init__(self):
        super(EmptyDataset, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset, no items to get")
    
class MagNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the MagNet dataset.

    Encapsulates all data loading logic, including:
    - Reading raw CSV data.
    - Preprocessing (interpolation, log transform, normalization).
    - Splitting into train/val/test sets.
    - Creating DataLoaders.
    """
    def __init__(self, data_B, data_F, data_T, data_P=None, norm_info_path=None,
                 batch_size=1, num_workers=1, sample_num=1024):
        """
        Initialize the DataModule.

        Args:
            data_B (pd.DataFrame): DataFrame containing B-field waveforms.
            data_F (pd.DataFrame): DataFrame containing Frequency values.
            data_T (pd.DataFrame): DataFrame containing Temperature values.
            data_P (pd.DataFrame, optional): DataFrame containing Power Loss (Ground Truth). Defaults to None.
            norm_info_path (str, optional): Path to a JSON file containing normalization statistics. Defaults to None.
            batch_size (int, optional): Batch size for dataloaders. Defaults to 1.
            num_workers (int, optional): Number of workers for dataloaders. Defaults to 1.
            sample_num (int, optional): Target number of samples for B-field interpolation. Defaults to 1024.
        """
        super().__init__()
        # data are loaded using pd.read_csv(in_file, header=None)
        self.data_B = data_B 
        self.data_F = data_F 
        self.data_T = data_T 
        self.data_P = data_P 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_num = sample_num # sample number for each data point.

        # Load normalization info if provided
        if not norm_info_path is None:
            with open(norm_info_path, 'r') as file:
                self.norm_info = json.load(file)
        else:
            self.norm_info = None

    def prepare_data(self):
        """
        Prepare data for training/inference.
        
        This method performs:
        1. Converting DataFrames to Tensors.
        2. Interpolating B-field to `sample_num` points.
        3. Log-transforming Frequency and Power Loss.
        4. Calculating normalization statistics (if not provided).
        5. Normalizing all inputs.
        """
        # Convert B-field data to tensor and reshape for interpolation
        in_B = torch.from_numpy(self.data_B.values).float().unsqueeze(2)
        
        # Downsample/Upsample B-field to fixed sample_num
        N, D, C = in_B.size()
        # Interpolate: (N, C, D) -> (N, C, sample_num) -> (N, sample_num, C)
        in_B = torch.nn.functional.interpolate(in_B.view(N, C, D), size=self.sample_num, mode='linear').view(N, -1, C)
        
        # Convert Temperature to tensor
        in_T = torch.from_numpy(self.data_T.values).float().view(-1, 1)

        # Convert Frequency to tensor
        in_F = self.data_F
        in_F = torch.from_numpy(in_F.values).float().view(-1, 1)

        # Transform Frequency: Apply Log10 (assumed natural log here based on torch.log)
        # Note: torch.log is natural log (ln). consistency check recommended if log10 was intended.
        in_F = torch.log(in_F)

        # Calculate Normalization Statistics if not loaded from file
        if self.norm_info == None:
            self.normB = [torch.mean(in_B), torch.std(in_B)]
            self.normF = [torch.mean(in_F), torch.std(in_F)]
            self.normT = [torch.mean(in_T), torch.std(in_T)]
        else:
            self.normB = self.norm_info['normB']
            self.normF = self.norm_info['normF']
            self.normT = self.norm_info['normT']

        # Normalize Inputs (Standard Scaling: (x - mean) / std)
        in_B = (in_B - self.normB[0]) / self.normB[1]
        in_F = (in_F - self.normF[0]) / self.normF[1]
        in_T = (in_T - self.normT[0]) / self.normT[1]

        # Handle Ground Truth Power Loss (data_P)
        if not self.data_P is None:
            gt_P = torch.from_numpy(self.data_P.values).float().view(-1, 1)
            out_P = torch.log(gt_P) # Log transform Power Loss
            self.normP = [torch.mean(out_P), torch.std(out_P)]
            out_P = (out_P - self.normP[0]) / self.normP[1] # Normalize Power Loss
        else:
            # fake ground truth for inference mode
            gt_P = torch.zeros((N, 1)) 
            out_P = torch.zeros((N, 1)) 
            assert self.norm_info != None, 'norm_info is nessesary when groundtruth is not provided!'
            self.normP = self.norm_info['normP']

        print('Log for sample dims:')
        print(in_B.size())  # torch.Size([40712, 1024, 1])
        print(in_T.size())  # torch.Size([40712, 1])
        print(in_F.size())  # torch.Size([40712, 1])
        print(out_P.size())  # torch.Size([40712, 1])
        print(gt_P.size())  # torch.Size([40712, 1])

        # Create TensorDataset
        self.dataset = TensorDataset(in_B, in_F, in_T, out_P, gt_P)

    def setup(self, stage, train_ratio=0.8, val_ratio=0.2):
        """
        Split the dataset into train, val, and test based on the stage.

        Args:
            stage (str): 'fit' for training/validation, 'inference' for testing.
            train_ratio (float, optional): Ratio of training data. Defaults to 0.8.
            val_ratio (float, optional): Ratio of validation data. Defaults to 0.2.
        """
        if stage == 'fit':
            train_size = int(train_ratio * len(self.dataset))
            valid_size = int(val_ratio * len(self.dataset))
            test_size = len(self.dataset) - train_size - valid_size
            (
                self.train_dataset,
                self.valid_dataset,
                self.test_dataset,
            ) = torch.utils.data.random_split(
                self.dataset, [train_size, valid_size, test_size]
            )
        elif stage == 'inference':
            # In inference mode, use the entire dataset for testing
            train_size, valid_size, test_size = 0, 0, len(self.dataset)
            
            self.train_dataset = EmptyDataset()
            self.valid_dataset = EmptyDataset()
            self.test_dataset = self.dataset

        print(rf'Split the dataset: Train({train_size}) | Val({valid_size}) | Test({test_size})')

    def train_dataloader(self):
        """Returns DataLoader for training data."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Returns DataLoader for validation data."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Returns DataLoader for test data."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)