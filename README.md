# Graph Wavelet Encoder

Graph wavelet scattering transform encoder for [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Computes multi-scale wavelet features from graph structure and node signals using lazy random-walk diffusion.

## Installation

From source (for now):

```bash
pip install -e .
```

With dev dependencies (pytest, etc.):

```bash
pip install -e ".[dev]"
```

## Usage

```python
import torch
from torch_geometric.data import Data, Batch
from graph_wavelet_encoder import GraphWaveletEncoder

# Single graph or batched PyG Data
# Expects: .x (node features), .edge_index, .batch (for batched graphs)
encoder = GraphWaveletEncoder(
    scales=(1, 2, 4, 8, 16),
    sigma=2.0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
features = encoder.encode(graph)  # [batch_size, num_nodes, num_features_per_node]
```

The encoder uses a lazy random-walk matrix and produces zeroth-, first-, and second-order scattering coefficients. See the docstrings in `graph_wavelet_encoder.encoder` for details.

## Development

- **Package layout:** [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) — package lives under `src/graph_wavelet_encoder/`.

### Testing and benchmarking

```bash
pytest tests/test_encoder.py -v -s
```

## License

Yale Licence
