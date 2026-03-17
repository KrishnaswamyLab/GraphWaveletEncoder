from typing import List
import numpy as np
from scipy import sparse
import torch
from torch_geometric.utils import unbatch, to_scipy_sparse_matrix


class GraphWaveletEncoder(object):
    '''
    Graph Wavelet Scattering Transform Encoder
    Given adjacency matrix and node signals, computes multi-scale wavelet features.
    '''
    def __init__(self, scales, sigma: float = 2.0, eps: float = 1e-8, device=torch.device('cpu')):
        self.scales = scales
        self.sigma = sigma
        self.device = device
        self.num_wavelets = len(scales)
        self.num_features_per_node = 1 + self.num_wavelets + (self.num_wavelets * (self.num_wavelets - 1)) // 2
        self.eps = eps

    def lazy_random_walk(self, graph) -> List[sparse.csr_matrix]:
        '''Convert adjacency to lazy random walk matrix P = 0.5(I + D^-1 A)'''

        num_graphs = int(graph.batch.max()) + 1 if graph.batch is not None else 1

        P_list = []
        for graph_idx in range(num_graphs):
            # Convert the graph to a sparse adjacency matrix.
            if graph.batch is not None:
                start, end = graph.ptr[graph_idx].item(), graph.ptr[graph_idx + 1].item()
                edge_mask = ((graph.edge_index[0] >= start) & (graph.edge_index[0] < end) & (graph.edge_index[1] >= start) & (graph.edge_index[1] < end))
                edge_index_sub = graph.edge_index[:, edge_mask] - start
                num_nodes_sub = end - start
                edge_weight_sub = None
                if graph.edge_attr is not None:
                    dist = graph.edge_attr[edge_mask][:, 0]
                    edge_weight_sub = torch.exp(-(dist ** 2) / (self.sigma ** 2)).detach().cpu()
                A = to_scipy_sparse_matrix(edge_index_sub, edge_attr=edge_weight_sub, num_nodes=num_nodes_sub).astype(np.float32)
            else:
                edge_weight = None
                if graph.edge_attr is not None:
                    dist = graph.edge_attr[:, 0]
                    edge_weight = torch.exp(-(dist ** 2) / (self.sigma ** 2)).detach().cpu()
                num_nodes = graph.x.shape[0]
                A = to_scipy_sparse_matrix(graph.edge_index, edge_attr=edge_weight, num_nodes=num_nodes).astype(np.float32)
            # Compute diagonal.
            D = np.array(A.sum(axis=1)).ravel()
            D_inv = np.reciprocal(np.clip(D, self.eps, None))
            # Diffusion matrix.
            P_t = sparse.diags(D_inv) @ A
            I = sparse.eye(P_t.shape[0], dtype=np.float32)
            # Lazy random walk matrix.
            P = 0.5 * (I + P_t)
            P_list.append(P)
        return P_list

    def graph_wavelet_transform(self, P_list: List[sparse.csr_matrix], X: np.ndarray) -> np.ndarray:
        '''
        Graph wavelet scattering transform.

        Wavelet at scale s:  W_s f = |P^s f - P^{2s} f|

        Features per node:
            F0 = X                                              (zeroth order)
            F1 = concat_s [ |W_s X| ]                           (first order)
            F2 = concat_{s_i > s_j} [ |W_{s_i}(|W_{s_j} X|)| ]  (second order)

        Output: np.ndarray [batch_size, N, F * (1 + S + S(S-1)/2)]
        '''

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        batch_size = len(P_list)
        assert batch_size == X.shape[0]
        # Need P^{2s} for largest s, so max power = 2 * max(scales).
        max_power = max(self.scales) * 2

        features = []
        for graph_idx in range(batch_size):
            P_batch = P_list[graph_idx]
            X_batch = X[graph_idx]  # [N, F]

            # X_powers[k] = P^k X, built iteratively: P^k X = P @ P^{k-1} X
            X_powers = [X_batch]
            for _ in range(max_power):
                X_powers.append(P_batch @ X_powers[-1])

            # First-order wavelets: W_s X = |P^s X - P^{2s} X|
            # Kept as a list for reuse in second-order computation.
            wavelets = [np.abs(X_powers[s] - X_powers[2 * s]) for s in self.scales]

            # Collect per-node features [N, F] into a single list, then concatenate.
            # All wavelet features are rescaled by log(1+s) to compensate for the
            # decay of diffusion coefficients at larger scales (P has eigenvalues in
            # [0,1], so P^s - P^{2s} shrinks with s).
            #
            #   F0 = X                                              (zeroth order)
            #   F1_s = |W_s X| * log(1+s)                           (first order)
            #   F2_{i,j} = |W_{s_i}(|W_{s_j} X|)| * log(1+s_i+s_j)  (second order, s_i > s_j)
            node_features = [X_batch]

            for W, s in zip(wavelets, self.scales):
                node_features.append(W * np.log1p(s))

            for i in range(1, len(self.scales)):
                for j in range(i):
                    # W_{s_i}(u) = |P^{s_i} u - P^{2 s_i} u|, where u = |W_{s_j} X|
                    u_s = apply_power(P_batch, wavelets[j], self.scales[i])
                    u_2s = apply_power(P_batch, u_s, self.scales[i])
                    s_eff = self.scales[i] + self.scales[j]
                    node_features.append(np.abs(u_s - u_2s) * np.log1p(s_eff))

            features.append(np.concatenate(node_features, axis=1))

        return np.stack(features, axis=0)

    def encode(self, graph):
        '''
        Returns:
            features: [B, N, num_features_per_node] wavelet features
        '''
        # Lazy random walk
        P_list = self.lazy_random_walk(graph)

        # Compute graph wavelets (use a local batch vector — never mutate the input)
        batch = graph.batch if graph.batch is not None else torch.zeros(
            graph.x.shape[0], dtype=torch.long, device=graph.x.device
        )
        X = torch.stack(unbatch(graph.x, batch))
        features = self.graph_wavelet_transform(P_list, X)

        # Single transfer to device (non_blocking to overlap with next ops).
        features = torch.from_numpy(features).float().to(self.device, non_blocking=True)

        return features


def apply_power(P, X_in, power: int):
    out = X_in
    for _ in range(power):
        out = P @ out
    return out
