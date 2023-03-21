import os
import torch

from typing import List

from torch_geometric.datasets import Reddit, Flickr
from torch_geometric.transforms import ToUndirected

from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.data.dataset import Dataset
from torch_geometric.utils import to_torch_coo_tensor, add_self_loops, scatter


# Local path to save diffusion operators to.
SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'diffusion_ops'
)


def get_device() -> str:
    # Get the device to move PyTorch tensors to.
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _diffuse_features(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        num_layers: int,
        verbose: bool) -> List[torch.Tensor]:

    if verbose:
        print('Computing symmetric normalised adjacency matrix...')

    # Compute the symmetric normalised adjacency matrix.
    num_nodes = features.size(0)
    sym_norm_adj = get_sym_norm_adj(edge_index, num_nodes)

    # Make sure no infs or NaNs made it through.
    assert not torch.any(torch.isinf(sym_norm_adj)) and not torch.any(torch.isnan(sym_norm_adj))

    if verbose:
        print('Done!\n')
        print('Diffusing node features...')

    # Compute the diffusion operators using sparse matrix multiplication.
    diffusion_ops = [features]
    for i in range(num_layers):
        diffusion_op = torch.sparse.mm(sym_norm_adj, diffusion_ops[-1])
        diffusion_ops.append(diffusion_op)
        assert not torch.any(torch.isinf(diffusion_op)) and not torch.any(torch.isnan(diffusion_op))

        if verbose:
            print(f'>>> Layer {i + 1}/{num_layers}')

    if verbose:
        print('Done!\n')

    return diffusion_ops


def get_sym_norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = get_device()
    edge_index, _ = add_self_loops(edge_index)

    # Compute the degree matrix.
    _, col = edge_index
    edge_weight = torch.ones(edge_index.size(1)).to(device)
    deg = scatter(edge_weight, col, 0, num_nodes, reduce='sum')

    # Compute D^{-1/2}
    values = torch.sqrt(1 / deg)
    indices = torch.arange(num_nodes)
    indices = torch.vstack((indices, indices)).to(device)
    D_inv_sqrt = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    # Get the adjacency matrix as a sparse COO tensor.
    A = to_torch_coo_tensor(edge_index)

    # Compute the symmetric normalised adjacency matrix.
    return D_inv_sqrt @ A @ D_inv_sqrt


def get_diffusion_ops(
        dataset: Dataset,
        num_layers: int = 4,
        verbose: bool = True) -> List[torch.Tensor]:

    file_name = dataset.__class__.__name__.lower()
    if hasattr(dataset, 'name'):
        file_name += '_' + dataset.name.lower()

    file_path = os.path.join(SAVE_PATH, f'{file_name}.pt')

    # Try to load existing diffusion operators if present.
    try:
        return torch.load(file_path, map_location=get_device())

    except FileNotFoundError:
        # If not found, compute them.
        data = dataset[0].to(get_device())
        diffusion_ops = _diffuse_features(data.x, data.edge_index, num_layers, verbose)

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        file_path = os.path.join(SAVE_PATH, f'{file_name}.pt')
        torch.save(diffusion_ops, file_path)

        return diffusion_ops


def _process_datasets():
    # Define the datasets to compute diffusion operators for.
    datasets = [
        Reddit(root='./data', transform=ToUndirected()),
        Flickr(root='./data', transform=ToUndirected()),
        PygNodePropPredDataset(
            root='./data',
            name='ogbn-arxiv',
            transform=ToUndirected()
        )
    ]

    for dataset in datasets:
        print('-' * 25 + dataset.__class__.__name__ + '-' * 25)
        get_diffusion_ops(dataset)


if __name__ == '__main__':
    _process_datasets()
