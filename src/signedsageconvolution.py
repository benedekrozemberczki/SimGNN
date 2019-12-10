"""Layer classes."""

import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class SignedSAGEConvolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SignedSAGEConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

class SignedSAGEConvolutionBase(SignedSAGEConvolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """
    def forward(self, x, edge_index):
        """
        Forward propagation pass with features an indices.
        :param x: Feature matrix.
        :param edge_index: Indices.
        """
        edge_index, _ = remove_self_loops(edge_index, None)
        row, col = edge_index

        if self.norm:
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        else:
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))

        out = torch.cat((out, x), 1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out

class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x_1, x_2, edge_index_pos, edge_index_neg):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index_pos: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        edge_index_pos, _ = remove_self_loops(edge_index_pos, None)
        edge_index_pos = add_self_loops(edge_index_pos, num_nodes=x_1.size(0))
        edge_index_neg, _ = remove_self_loops(edge_index_neg, None)
        edge_index_neg = add_self_loops(edge_index_neg, num_nodes=x_2.size(0))

        row_pos, col_pos = edge_index_pos
        row_neg, col_neg = edge_index_neg

        if self.norm:
            out_1 = scatter_mean(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_mean(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))
        else:
            out_1 = scatter_add(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_add(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))

        out = torch.cat((out_1, out_2, x_1), 1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out
