# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from .compressor import Compressor
import pickle
import sys
import logging

class QSGDCompressor(Compressor):
    """Quantization compressor. 
    
    A implementation for paper https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf.
    
    Alistarh, Dan, et al. "QSGD: Communication-efficient SGD via gradient quantization and encoding." Advances in Neural Information Processing Systems 30 (2017): 1709-1720.
    Thanks to git repo: https://github.com/xinyandai/gradient-quantization
    
    Args:
        n_bit (int): the bits num for quantization. Bigger n_bit comes with better compress precision but more communication consumption.
        random (bool, optional): Carry bit with probability. Defaults to True.
        cuda (bool, optional): use GPU. Defaults to False.
    """
    def __init__(self, random=True, cuda=False):
        self.random = random

        self.cuda = cuda
        self.s = None
        self.bit = None


    def compress(self, tensor, n_bit=8, packing=False):
        """Compress a tensor with quantization
        Args:
            tensor ([type]): [description]
        Returns:
            norm (torch.Tensor): The normalization number.
            signs (torch.Tensor): Tensor that indicates the sign of coresponding number.
            quantized_intervals (torch.Tensor): Quantized tensor that each item in [0, 2**n_bit -1].
        """
        try:
            #covert tensor to cpu if in cuda
            self.bit = n_bit
            self.s = 2**self.bit
            if packing:
                self.code_dtype = getattr(torch, f"int{n_bit}")  # Use n_bit dtype for l
            else:
                self.code_dtype = torch.float32
            if type(tensor) != torch.float32:
                # logging.info(f'convert tensor from {type(tensor)} to torch.float32')
                tensor = torch.tensor(tensor, dtype=torch.float32)
            shape = tensor.shape
            vec = tensor.view(-1)
            # norm = torch.norm(vec, dim=1, keepdim=True)
            norm = torch.max(torch.abs(vec), dim=0, keepdim=True)[0]
            normalized_vec = vec / norm

            scaled_vec = torch.abs(normalized_vec) * self.s
            l = torch.clamp(scaled_vec, 0, self.s - 1).type(self.code_dtype)

            if self.random:
                # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
                probabilities = scaled_vec - l.type(torch.float32)
                r = torch.rand(l.size())
                if self.cuda:
                    r = r.cuda()
                l[:] += (probabilities > r).type(self.code_dtype)

            signs = torch.sign(vec) > 0
            size = sys.getsizeof(pickle.dumps([norm, signs.view(shape), l.view(shape)]))
            return [norm, signs.view(shape), l.view(shape)], size
        except Exception as e:
            logging.info(f"Compress error: {e}")

    def decompress(self, signature, n_bit=8, packing=False):
        """Decompress tensor
        Args:
            signature (list): [norm, signs, quantized_intervals], returned by :func:``compress``.
        Returns:
            torch.Tensor: Raw tensor represented by signature.
        """
        self.bit = n_bit
        self.s = 2**self.bit
        if packing:
            self.code_dtype = getattr(torch, f"int{n_bit}")  # Use n_bit dtype for l
        else:
            self.code_dtype = torch.float32
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        shape = l.shape
        scaled_vec = l.type(
            torch.float32) * (2 * signs.type(torch.float32) - 1)
        compressed = (scaled_vec.view((-1))) * norm / self.s
        return compressed.view(shape)