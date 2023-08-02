import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.utils.prune as prune
import time
import numpy as np
import warnings
from collections import OrderedDict, defaultdict
import copy
import logging


class LinearMasked(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(LinearMasked, self).__init__(in_features, out_features, bias)
            self.mask_flag = False

        def set_mask(self, mask):
            self.mask = Variable(mask, requires_grad=False, volatile=False)
            self.weight.data = self.weight.data * self.mask.data
            self.mask_flag = True

        def get_mask(self):
            logging.info(self.mask_flag)
            return self.mask

        def forward(self, x):
            if self.mask_flag:
                weight = self.weight * self.mask
                return F.linear(x, weight, self.bias)
            else:
                return F.linear(x, self.weight, self.bias)


class Conv2dMasked(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dMasked, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = Variable(mask, requires_grad=False, volatile=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        logging.info(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.conv2d(
                x,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
                
class Pruning():
    def __init__(self, cuda=False):
        warnings.filterwarnings("ignore")
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.vanilla_training_time = 0
    
    def nonzero(self, tensor):
        """Returns absolute number of values different from 0

        Arguments:
            tensor {numpy.ndarray} -- Array to compute over

        Returns:
            int -- Number of nonzero elements
        """
        return np.sum(tensor != 0.0)


    # https://pytorch.org/docs/stable/tensor_attributes.html
    dtype2bits = {
        torch.float32: 32,
        torch.float: 32,
        torch.float64: 64,
        torch.double: 64,
        torch.float16: 16,
        torch.half: 16,
        torch.uint8: 8,
        torch.int8: 8,
        torch.int16: 16,
        torch.short: 16,
        torch.int32: 32,
        torch.int: 32,
        torch.int64: 64,
        torch.long: 64,
        torch.bool: 1,
    }

    def hook_applyfn(self, hook, model, forward=False, backward=False):
        """

        [description]

        Arguments:
            hook {[type]} -- [description]
            model {[type]} -- [description]

        Keyword Arguments:
            forward {bool} -- [description] (default: {False})
            backward {bool} -- [description] (default: {False})

        Returns:
            [type] -- [description]
        """
        assert forward ^ backward, "Either forward or backward must be True"
        hooks = []

        def register_hook(module):
            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not isinstance(module, nn.ModuleDict)
                and not (module == model)
            ):
                if forward:
                    hooks.append(module.register_forward_hook(hook))
                if backward:
                    hooks.append(module.register_backward_hook(hook))

        return register_hook, hooks

    def get_activations(self, model, input):

        activations = OrderedDict()

        def store_activations(module, input, output):
            if isinstance(module, nn.ReLU):
                # TODO ResNet18 implementation reuses a
                # single ReLU layer?
                return
            assert module not in activations, f"{module} already in activations"
            # TODO [0] means first input, not all models have a single input
            
            activations[module] = (
                input[0].detach().cpu().numpy().copy(),
                output.detach().cpu().numpy().copy(),
            )

        fn, hooks = self.hook_applyfn(store_activations, model, forward=True)
        # Ensure input and model are on the same device
        input = input.to(self.device)
        model = model.to(self.device)
        model.apply(fn)
        with torch.no_grad():
            model(input)

        for h in hooks:
            h.remove()

        return activations

    def dense_flops(self, in_neurons, out_neurons):
        """Compute the number of multiply-adds used by a Dense (Linear) layer"""
        return in_neurons * out_neurons


    def conv2d_flops(self, 
        in_channels,
        out_channels,
        input_shape,
        kernel_shape,
        padding="same",
        strides=1,
        dilation=1,
    ):
        """Compute the number of multiply-adds used by a Conv2D layer
        Args:
            in_channels (int): The number of channels in the layer's input
            out_channels (int): The number of channels in the layer's output
            input_shape (int, int): The spatial shape of the rank-3 input tensor
            kernel_shape (int, int): The spatial shape of the rank-4 kernel
            padding ({'same', 'valid'}): The padding used by the convolution
            strides (int) or (int, int): The spatial stride of the convolution;
                two numbers may be specified if it's different for the x and y axes
            dilation (int): Must be 1 for now.
        Returns:
            int: The number of multiply-adds a direct convolution would require
            (i.e., no FFT, no Winograd, etc)
        # >>> c_in, c_out = 10, 10
        # >>> in_shape = (4, 5)
        # >>> filt_shape = (3, 2)
        # >>> # valid padding
        # >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, padding='valid')
        # >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 4))
        # True
        # >>> # same padding, no stride
        # >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, padding='same')
        # >>> ret == int(c_in * c_out * np.prod(filt_shape) * np.prod(in_shape))
        # True
        # >>> # valid padding, stride > 1
        # >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, \
        #                 padding='valid', strides=(1, 2))
        # >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 2))
        # True
        # >>> # same padding, stride > 1
        # >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, \
        #                     padding='same', strides=2)
        # >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 3))
        True
        """
        # validate + sanitize input
        assert in_channels > 0
        assert out_channels > 0
        assert len(input_shape) == 2
        assert len(kernel_shape) == 2
        padding = padding.lower()
        assert padding in (
            "same",
            "valid",
            "zeros",
        ), "Padding must be one of same|valid|zeros"
        try:
            strides = tuple(strides)
        except TypeError:
            # if one number provided, make it a 2-tuple
            strides = (strides, strides)
        assert dilation == 1 or all(
            d == 1 for d in dilation
        ), "Dilation > 1 is not supported"

        # compute output spatial shape
        # based on TF computations https://stackoverflow.com/a/37674568
        if padding in ["same", "zeros"]:
            out_nrows = np.ceil(float(input_shape[0]) / strides[0])
            out_ncols = np.ceil(float(input_shape[1]) / strides[1])
        else:  # padding == 'valid'
            out_nrows = np.ceil((input_shape[0] - kernel_shape[0] + 1) / strides[0])  # noqa
            out_ncols = np.ceil((input_shape[1] - kernel_shape[1] + 1) / strides[1])  # noqa
        output_shape = (int(out_nrows), int(out_ncols))

        # work to compute one output spatial position
        nflops = in_channels * out_channels * int(np.prod(kernel_shape))

        # total work = work per output position * number of output positions
        return nflops * int(np.prod(output_shape))

    def _conv2d_flops(self, module, activation):
        # Auxiliary func to use abstract flop computation

        # Drop batch & channels. Channels can be dropped since
        # unlike shape they have to match to in_channels
        input_shape = activation.shape[2:]
        # TODO Add support for dilation and padding size
        return self.conv2d_flops(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            input_shape=input_shape,
            kernel_shape=module.kernel_size,
            padding=module.padding_mode,
            strides=module.stride,
            dilation=module.dilation,
        )


    def _linear_flops(self, module, activation):
        # Auxiliary func to use abstract flop computation
        return self.dense_flops(module.in_features, module.out_features)


    def flops(self, model, input):
        """Compute Multiply-add FLOPs estimate from model

        Arguments:
            model {torch.nn.Module} -- Module to compute flops for
            input {torch.Tensor} -- Input tensor needed for activations

        Returns:
            tuple:
            - int - Number of total FLOPs
            - int - Number of FLOPs related to nonzero parameters
        """
        FLOP_fn = {
            nn.Conv2d: self._conv2d_flops,
            nn.Linear: self._linear_flops,
            Conv2dMasked: self._conv2d_flops,
            LinearMasked: self._linear_flops,
        }
        total_flops = nonzero_flops = 0
        activations = self.get_activations(model, input)

        # The ones we need for backprop
        for m, (act, _) in activations.items():
            if m.__class__ in FLOP_fn:
                w = m.weight.detach().cpu().numpy().copy()
                module_flops = FLOP_fn[m.__class__](m, act)
                total_flops += module_flops
                # For our operations, all weights are symmetric so we can just
                # do simple rule of three for the estimation
                nonzero_flops += module_flops * self.nonzero(w).sum() / np.prod(w.shape)

        return total_flops, nonzero_flops

    def measure_module_sparsity(self, module, weight=True, bias=False, use_mask=False):

        num_zeros = 0
        num_elements = 0

        if use_mask == True:
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name and weight == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
                if "bias_mask" in buffer_name and bias == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
        else:
            for param_name, param in module.named_parameters():
                if "weight" in param_name and weight == True:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()
                if "bias" in param_name and bias == True:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity


    def measure_global_sparsity(self, model,
                                weight=True,
                                bias=False,
                                conv2d_use_mask=False,
                                linear_use_mask=False):

        num_zeros = 0
        num_elements = 0

        for module_name, module in model.named_modules():

            if isinstance(module, torch.nn.Conv2d):

                module_num_zeros, module_num_elements, _ = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

            elif isinstance(module, torch.nn.Linear):

                module_num_zeros, module_num_elements, _ = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=linear_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    def prune_model(self, model, prune_amount, trainloader):
        '''Prune the model by prune_amount'''
        x, _ = next(iter(trainloader))
        # logging.info(f'trainloader: {trainloader.test_data}')
        # logging.info('input shape: ', x.shape)
        start_time = time.time()
        parameters_to_prune = []
        
        pruned_model = copy.deepcopy(model)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
                
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_amount
        )

        elapsed_time = time.time() - start_time
        FLOPS, compression_ratio = self.calculate_prune_metrics(
        model, trainloader, self.device)
        
        # logging.info('FLOPS, compression ratio:', FLOPS, compression_ratio)
        logging.info(f'Pruning Overhead: {elapsed_time}')
        reduction_ratio = (FLOPS[1]/FLOPS[0])
        
        # Remove the pruned filters (channels)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.remove(module, 'weight')
                
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                pruned_weight = model.state_dict()[name + ".weight"]
                module.weight = nn.Parameter(pruned_weight)
        
        # logging.info(f"Prune Amount: {prune_amount * 100}%")
        logging.info(f"Pruning time Time: {elapsed_time} seconds")
        
        return pruned_model, reduction_ratio


    def model_size(self, model, as_bits=False):
        """Returns absolute and nonzero model size

        Arguments:
            model {torch.nn.Module} -- Network to compute model size over

        Keyword Arguments:
            as_bits {bool} -- Whether to account for the size of dtype

        Returns:
            int -- Total number of weight & bias params
            int -- Out total_params exactly how many are nonzero
        """

        total_params = 0
        nonzero_params = 0
        for tensor in model.parameters():
            t = np.prod(tensor.shape)
            nz = self.nonzero(tensor.detach().cpu().numpy())
            if as_bits:
                bits = self.dtype2bits[tensor.dtype]
                t *= bits
                nz *= bits
            total_params += t
            nonzero_params += nz
        return int(total_params), int(nonzero_params)

    def calculate_prune_metrics(self, net, test_loader, device):

        x, _ = next(iter(test_loader))
        # logging.info('input shape: ', x)
        x = x.to(device)

        size, size_nz = self.model_size(net)

        FLOPS = self.flops(net, x)
        compression_ratio = size / size_nz

        return FLOPS, compression_ratio
