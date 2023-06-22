from typing import List

import numpy as np
import torch

from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.internal.model_adapter_base import ModelAdapterBase


class TorchModelAdapter(ModelAdapterBase):
    """
    Adapts functions to pytorch models.
    """
    def __init__(self, model: torch.nn.Module, optimizer: TorchServerOptimizer = None):
        """
        Initializes a TorchModelAdapter.
        :param model: the PyTorch model to adapt
        :param optimizer: the optimizer to apply weights, when specified.
        """
        self.model = model
        self.optimizer = optimizer

    def set_weights(self, weights: List[np.ndarray]):
        """
        Set the model's weights to the numpy weights array.
        :param weights: numpy weights array
        """
        current_grad_weights = [param.data.clone() for param in self.model.state_dict().values()]
        new_state_dict = {
            name: torch.from_numpy(np.asarray(weights[i], dtype=np.float32))
            for i, name in enumerate(self.model.state_dict().keys())
        }
        self.model.load_state_dict(new_state_dict)
        if self.optimizer:
            self.optimizer.update_round_gradient(weights, current_grad_weights, self.model)

    def get_weights(self) -> List[np.ndarray]:
        """
        Get the model's weights as a numpy weights array. Note that it doesn't contain layer names. Rather, index 0
        contains the model's first layer weights, and index N contains the N+1 layer's weights.
        :return: A numpy array
        """
        return [params.data.clone() for params in self.model.state_dict().values()]

    def get_model(self):
        """
        Get the instantiated framework specific model including the architecture.
        """
        return self.model
