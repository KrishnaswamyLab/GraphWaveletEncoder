<div align="center">

# Graph Wavelet Encoder

[![Twitter Follow](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://x.com/KrishnaswamyLab)
[![Twitter](https://img.shields.io/twitter/follow/ChenLiu-1996.svg?style=social)](https://twitter.com/ChenLiu_1996)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ChenLiu-1996?color=blue)](https://www.linkedin.com/in/chenliu1996/)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Chen-4a86cf?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=3rDjnykAAAAJ&sortby=pubdate)
<br>
[![Latest PyPI version](https://img.shields.io/pypi/v/graph-wavelet-encoder.svg)](https://pypi.org/project/graph-wavelet-encoder/)
[![PyPI download 3 month](https://static.pepy.tech/badge/graph-wavelet-encoder)](https://pepy.tech/projects/graph-wavelet-encoder)
[![PyPI download month](https://img.shields.io/pypi/dm/graph-wavelet-encoder.svg)](https://pypistats.org/packages/graph-wavelet-encoder)

</div>

Graph wavelet scattering transform encoder for [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) graphs.

Computes multi-scale wavelet features from graph structure and node signals using lazy random-walk diffusion.

## Installation

```bash
pip install graph-wavelet-encoder
```

For development (editable install):

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
