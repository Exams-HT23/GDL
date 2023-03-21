import torch
import torch.nn as nn

from typing import List
from preprocessing import get_sym_norm_adj

from torch_geometric.nn import GCNConv
from torch.nn.functional import log_softmax


def get_device() -> str:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2) -> None:

        super(MLP, self).__init__()

        # Input layer.
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]

        # Hidden layers.
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

        # Output layer.
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layers(X)
        return log_softmax(X, dim=1)


class GNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1) -> None:

        # Need at least one decoder layer.
        super(GNN, self).__init__()
        assert num_decoder_layers > 0

        # Define the encoder part of the architecture.
        if num_encoder_layers > 0:
            encoder_layers = [nn.Linear(input_dim, hidden_dim)]

            for _ in range(num_encoder_layers - 1):
                encoder_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ])

            self.encoder = nn.Sequential(*encoder_layers)

        else:
            # No encoder if zero layers are specified.
            self.encoder = None

        # Define the decoder part of the architecture.
        decoder_layers = []
        for _ in range(num_decoder_layers - 1):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

        decoder_layers.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)


class GCN(GNN):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_gcn_layers: int = 2,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1) -> None:

        # Call the parent constructor.
        super(GCN, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_encoder_layers,
            num_decoder_layers
        )

        # Define the GCN layers.
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.extend([
                GCNConv(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Pass the input through the encoder.
        X = self.encoder(X)

        # Pass the result through the GCN layers.
        for layer in self.gcn_layers:
            if isinstance(layer, GCNConv):
                X = layer(X, edge_index)
            else:
                X = layer(X)

        # Pass the result from the GCN layers through the decoder.
        X = self.decoder(X)

        # Get probabilistic outputs for each class.
        return log_softmax(X, dim=1)


class GRAFF(GNN):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1) -> None:

        # Call the parent constructor.
        super(GRAFF, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_encoder_layers,
            num_decoder_layers
        )

        # Define the weight matrix to be shared across all GRAFF layers.
        self.device = get_device()
        self.weights = nn.Parameter(torch.empty(hidden_dim, hidden_dim)).to(self.device)
        nn.init.xavier_uniform_(self.weights)

    def _get_sym_weights(self) -> torch.Tensor:
        # Convert the weight matrix to symmetric form.
        return self.weights.triu() + self.weights.triu(1).transpose(0, 1)


class StandardGRAFF(GRAFF):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            edge_index: torch.Tensor,
            num_nodes: int,
            num_graff_layers: int = 2,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1,
            use_encoder: bool = True) -> None:

        # The hidden dimension must match the input dimension if no encoder is used.
        if not use_encoder:
            assert input_dim == hidden_dim

        # Call the parent constructor.
        super(StandardGRAFF, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_encoder_layers,
            num_decoder_layers
        )

        # Get the symmetric normalised adjacency matrix.
        self.use_encoder = use_encoder
        self.num_layers = num_graff_layers
        self.sym_norm_adj = get_sym_norm_adj(edge_index, num_nodes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Pass the input through the encoder.
        if self.use_encoder:
            X = self.encoder(X)

        # Pass the result through the GRAFF layers.
        weights_sym = self._get_sym_weights()
        for _ in range(self.num_layers):
            X = X + self.sym_norm_adj @ X @ weights_sym

        # Pass the result from the GCN layers through the decoder.
        X = self.decoder(X)

        # Get probabilistic outputs for each class.
        return log_softmax(X, dim=1)


class ScalableGRAFF(GRAFF):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            diffusion_ops: List[torch.Tensor],
            num_graff_layers: int = 2,
            num_decoder_layers: int = 1) -> None:

        # Call the parent constructor.
        super(ScalableGRAFF, self).__init__(
            input_dim,
            input_dim,
            output_dim,
            0,  # No encoder for scalable GRAFF.
            num_decoder_layers
        )

        # Add one as the first "layer" is just X.
        num_graff_layers += 1

        # Make sure enough diffusion operators were given.
        assert len(diffusion_ops) >= num_graff_layers
        self.diffusion_ops = diffusion_ops[:num_graff_layers]

        # Get binomial coefficients.
        self.coefficients = ScalableGRAFF.__get_coefficents(num_graff_layers)

    @staticmethod
    def __get_coefficents(num_layers: int) -> List[int]:
        if num_layers == 0:
            return [1]

        # Recursively compute binomial coefficients.
        prev = ScalableGRAFF.__get_coefficents(num_layers - 1)
        coefficients = [prev[i - 1] + prev[i] for i in range(1, len(prev))]
        return [1] + coefficients + [1]

    def forward(self, _) -> torch.Tensor:
        # Get the symmetric weight matrix.
        weights_sym = self._get_sym_weights()
        weights_sym_power = torch.eye(weights_sym.shape[0]).to(self.device)

        # Pass the result through the GRAFF layers.
        X = torch.clone(self.diffusion_ops[0])
        for coefficient, diffusion_op in zip(self.coefficients, self.diffusion_ops[1:]):
            weights_sym_power = torch.mm(weights_sym_power, weights_sym)
            X += coefficient * torch.mm(diffusion_op, weights_sym_power)

        # Pass the result from the GCN layers through the decoder.
        X = self.decoder(X)

        # Get probabilistic outputs for each class.
        return log_softmax(X, dim=1)
