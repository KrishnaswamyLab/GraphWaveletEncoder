import math
from typing import List

import torch
from torch_geometric.utils import unbatch


class GraphWaveletEncoder(object):
    '''
    Graph Wavelet Scattering Transform Encoder
    Given adjacency matrix and node signals, computes multi-scale wavelet features.

    Not a torch.nn.Module (there are no learnable parameters), but exposes
    compatible APIs: __call__, forward, and to(device).
    '''
    def __init__(self, scales, sigma: float = 2.0, eps: float = 1e-8, device=torch.device('cpu')):
        self.scales = scales
        self.sigma = sigma
        self.device = device
        self.num_wavelets = len(scales)
        self.num_features_per_node = 1 + self.num_wavelets + (self.num_wavelets * (self.num_wavelets - 1)) // 2
        self.eps = eps

    def lazy_random_walk(self, graph) -> List[torch.Tensor]:
        '''Convert adjacency to lazy random walk matrix P = 0.5(I + D^-1 A).

        Returns a list of torch sparse CSR tensors on self.device.
        '''
        num_graphs = int(graph.batch.max()) + 1 if graph.batch is not None else 1
        device = self.device

        P_list: List[torch.Tensor] = []
        for graph_idx in range(num_graphs):
            if graph.batch is not None:
                start, end = graph.ptr[graph_idx].item(), graph.ptr[graph_idx + 1].item()
                edge_mask = (
                    (graph.edge_index[0] >= start) & (graph.edge_index[0] < end) &
                    (graph.edge_index[1] >= start) & (graph.edge_index[1] < end)
                )
                row = (graph.edge_index[0, edge_mask] - start).to(device)
                col = (graph.edge_index[1, edge_mask] - start).to(device)
                N = end - start
                if graph.edge_attr is not None:
                    dist = graph.edge_attr[edge_mask][:, 0].to(device)
                    values = torch.exp(-(dist ** 2) / (self.sigma ** 2))
                else:
                    values = torch.ones(row.shape[0], dtype=torch.float32, device=device)
            else:
                N = graph.x.shape[0]
                row = graph.edge_index[0].to(device)
                col = graph.edge_index[1].to(device)
                if graph.edge_attr is not None:
                    dist = graph.edge_attr[:, 0].to(device)
                    values = torch.exp(-(dist ** 2) / (self.sigma ** 2))
                else:
                    values = torch.ones(row.shape[0], dtype=torch.float32, device=device)

            D = torch.zeros(N, dtype=torch.float32, device=device)
            if row.numel() > 0:
                D.scatter_add_(0, row, values)
            D_inv = 1.0 / D.clamp(min=self.eps)

            # P = 0.5 * (I + D^{-1} A): off-diag from 0.5*D^{-1}*A, diag from 0.5*I
            # coalesce() sums duplicates so self-loops in A are handled correctly.
            scaled = 0.5 * values * D_inv[row]
            diag_idx = torch.arange(N, device=device)
            all_row = torch.cat([row, diag_idx])
            all_col = torch.cat([col, diag_idx])
            all_val = torch.cat([scaled, torch.full((N,), 0.5, device=device)])

            P = torch.sparse_coo_tensor(
                torch.stack([all_row, all_col]), all_val, (N, N)
            ).coalesce().to_sparse_csr()
            P_list.append(P)

        return P_list

    def graph_wavelet_transform(self, P_list: List[torch.Tensor], X: torch.Tensor) -> torch.Tensor:
        '''
        Graph wavelet scattering transform.

        Wavelet at scale s:  W_s f = |P^s f - P^{2s} f|

        Features per node:
            F0 = X                                              (zeroth order)
            F1 = concat_s [ |W_s X| ]                           (first order)
            F2 = concat_{s_i > s_j} [ |W_{s_i}(|W_{s_j} X|)| ]  (second order)

        F0 and F1 are always present. F2 requires at least two scales; passing a
        single-element `scales` list effectively disables second-order features.

        Returns: torch.Tensor [batch_size, N, F * (1 + S + S(S-1)/2)]
        '''
        batch_size = len(P_list)
        assert batch_size == X.shape[0]
        # Need P^{2s} for largest s, so max power = 2 * max(scales).
        max_power = max(self.scales) * 2

        features = []
        for graph_idx in range(batch_size):
            P = P_list[graph_idx]
            X_batch = X[graph_idx]  # [N, F]

            # X_powers[k] = P^k X, built iteratively: P^k X = P @ P^{k-1} X
            X_powers = [X_batch]
            for _ in range(max_power):
                X_powers.append(torch.sparse.mm(P, X_powers[-1]))

            # First-order wavelets: W_s X = |P^s X - P^{2s} X|
            wavelets = [torch.abs(X_powers[s] - X_powers[2 * s]) for s in self.scales]

            # All wavelet features are rescaled by log(1+·) to compensate for the
            # decay of diffusion coefficients at larger scales (P has eigenvalues in
            # [0,1], so P^s - P^{2s} shrinks with s).

            # F0: raw node features (1 block of F columns)
            node_features = [X_batch]

            # F1: first-order scattering (S blocks of F columns)
            for W, s in zip(wavelets, self.scales):
                node_features.append(W * math.log1p(s))

            # F2: second-order scattering (S(S-1)/2 blocks of F columns)
            for i in range(1, len(self.scales)):
                for j in range(i):
                    # W_{s_i}(u) = |P^{s_i} u - P^{2 s_i} u|, where u = |W_{s_j} X|
                    u_s = _apply_power(P, wavelets[j], self.scales[i])
                    u_2s = _apply_power(P, u_s, self.scales[i])
                    s_eff = self.scales[i] + self.scales[j]
                    node_features.append(torch.abs(u_s - u_2s) * math.log1p(s_eff))

            features.append(torch.cat(node_features, dim=1))

        return torch.stack(features, dim=0)

    def encode(self, graph):
        '''
        Returns:
            features: [B, N, num_features_per_node] wavelet features
        '''
        P_list = self.lazy_random_walk(graph)

        batch = graph.batch if graph.batch is not None else torch.zeros(
            graph.x.shape[0], dtype=torch.long, device=graph.x.device
        )
        X = torch.stack(unbatch(graph.x, batch)).to(self.device)
        features = self.graph_wavelet_transform(P_list, X)

        return features

    def forward(self, graph):
        '''Alias for encode(), matching PyTorch's forward convention.'''
        return self.encode(graph)

    def __call__(self, graph):
        '''Delegate to forward() so encoder(graph) works like a PyTorch module.'''
        return self.forward(graph)

    def to(self, device):
        '''Move the encoder to device (str, torch.device, or None).

        Returns self for chaining: encoder = GraphWaveletEncoder(scales).to('cuda').
        '''
        if device is not None:
            self.device = torch.device(device)
        return self


def _apply_power(P: torch.Tensor, X_in: torch.Tensor, power: int) -> torch.Tensor:
    out = X_in
    for _ in range(power):
        out = torch.sparse.mm(P, out)
    return out
