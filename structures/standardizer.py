import numpy as np
import torch
from typing import Union


class Standardizer:
    def __init__(self, input_dim: int, upper_bound: float, lower_bound: float, device):
        """

        :param input_dim:
        """

        self.device = device
        self.x_squares_sum = np.zeros(shape=(input_dim, ))
        self.x_sum = np.zeros(shape=(input_dim, ))
        self.n = 0

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.mean = None
        self.sigma = None

    def update_params(self):
        self.mean = self.x_sum/self.n
        self.sigma = np.square((self.x_squares_sum + self.n*self.mean**2 - 2*self.mean*self.x_sum)/self.n)
        self.mean_tensor = torch.from_numpy(self.mean).to(self.device, dtype=torch.float)
        self.sigma_tensor = torch.from_numpy(self.sigma).to(self.device, dtype=torch.float)

    def add_data(self, data: np.ndarray):
        self.x_squares_sum = np.add(self.x_squares_sum, data**2)
        self.x_sum = np.add(self.x_sum, data)
        self.n += 1

    def standarize(self, x: Union[np.ndarray, torch.Tensor]):
        if self.mean is None:
            return x  # not fitted yet

        if type(x) is np.ndarray:
            return np.clip((x-self.mean)/self.sigma, a_min=self.lower_bound, a_max=self.upper_bound)
        elif type(x) is torch.Tensor:
            return torch.clip((x-self.mean_tensor)/self.sigma_tensor, min=self.lower_bound, max=self.upper_bound)
        else:
            raise ValueError('Not a np array or a tensor')
