"""
Tests for GraphWaveletEncoder: correctness and throughput benchmarks.

    pytest tests/test_encoder.py -v -s
"""
import gc
import resource
import time

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_wavelet_encoder import GraphWaveletEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(num_nodes: int, num_edges: int = None, seed: int = 42):
    """Create a single graph with random topology."""
    if num_edges is None:
        num_edges = min(num_nodes * 3, num_nodes * (num_nodes - 1) // 2)
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, 1, generator=rng)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=rng)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.shape[1] == 0:
        edge_index = torch.randint(0, num_nodes, (2, 1), generator=rng)
    return Data(x=x, edge_index=edge_index)


def _make_batch(batch_size: int, num_nodes: int, num_edges_per_graph: int = None, seed: int = 0):
    """Create a batch of graphs with the same number of nodes (required by encoder)."""
    graphs = [
        _make_graph(num_nodes, num_edges_per_graph, seed=seed + i)
        for i in range(batch_size)
    ]
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestFeatureCount:
    """num_features_per_node must equal 1 + S + S*(S-1)//2 and match actual output."""

    @pytest.mark.parametrize("scales", [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16)])
    def test_feature_count_formula(self, scales):
        encoder = GraphWaveletEncoder(scales=scales, device=torch.device("cpu"))
        S = len(scales)
        expected = 1 + S + (S * (S - 1)) // 2
        assert encoder.num_features_per_node == expected

    @pytest.mark.parametrize("scales", [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16)])
    def test_output_shape_matches_feature_count(self, scales):
        """Actual output width must agree with num_features_per_node for all scale configs."""
        encoder = GraphWaveletEncoder(scales=scales, device=torch.device("cpu"))
        data = _make_graph(15, 30)
        out = encoder.encode(data)
        assert out.shape == (1, 15, encoder.num_features_per_node)


class TestSingleGraph:
    """Single-graph encoding."""

    def test_output_shape_dtype_device(self):
        num_nodes = 20
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        data = _make_graph(num_nodes)
        out = encoder.encode(data)
        assert out.shape == (1, num_nodes, encoder.num_features_per_node)
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"

    def test_with_edge_attr(self):
        num_nodes, num_edges = 10, 25
        rng = torch.Generator().manual_seed(1)
        x = torch.randn(num_nodes, 1, generator=rng)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=rng)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.rand(edge_index.shape[1], 1, generator=rng)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        encoder = GraphWaveletEncoder(scales=(1, 2), sigma=2.0, device=torch.device("cpu"))
        out = encoder.encode(data)
        assert out.shape == (1, num_nodes, encoder.num_features_per_node)

    def test_encode_does_not_mutate_input(self):
        """encode() must not modify the input graph object."""
        data = _make_graph(30, seed=123)
        assert data.batch is None
        encoder = GraphWaveletEncoder(scales=(1, 2, 4), device=torch.device("cpu"))
        encoder.encode(data)
        assert data.batch is None, "encode() should not set graph.batch"

    def test_deterministic_same_object(self):
        """Encoding the exact same Data object twice must give identical results."""
        data = _make_graph(30, seed=123)
        encoder = GraphWaveletEncoder(scales=(1, 2, 4), device=torch.device("cpu"))
        out1 = encoder.encode(data)
        out2 = encoder.encode(data)
        torch.testing.assert_close(out1, out2)

    def test_zeroth_order_equals_input(self):
        """F0 = X, so the first feature channel must equal the raw node signal."""
        data = _make_graph(10, seed=0)
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        out = encoder.encode(data)
        np_out = out.numpy()
        f0 = np_out[0, :, :1]
        x_np = data.x.numpy()
        np.testing.assert_allclose(f0, x_np, rtol=1e-5, atol=1e-5)

    def test_first_order_non_negative(self):
        """First-order wavelets |P^s X - P^{2s} X| must be non-negative."""
        data = _make_graph(20, seed=5)
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        out = encoder.encode(data)
        S = len(encoder.scales)
        first_order = out[0, :, 1 : 1 + S]
        assert (first_order >= -1e-6).all(), "First-order coefficients should be non-negative"

    def test_second_order_non_negative(self):
        """Second-order coefficients |W_{s_i}(|W_{s_j} X|)| must be non-negative."""
        data = _make_graph(20, seed=5)
        scales = (1, 2, 4)
        encoder = GraphWaveletEncoder(scales=scales, device=torch.device("cpu"))
        out = encoder.encode(data)
        S = len(scales)
        second_order = out[0, :, 1 + S:]
        assert second_order.shape[1] == (S * (S - 1)) // 2
        assert (second_order >= -1e-6).all(), "Second-order coefficients should be non-negative"

    def test_single_node_graph(self):
        """Graph with one node (no edges): encoder should still run."""
        data = Data(x=torch.randn(1, 1), edge_index=torch.zeros(2, 0, dtype=torch.long))
        encoder = GraphWaveletEncoder(scales=(1,), device=torch.device("cpu"))
        out = encoder.encode(data)
        assert out.shape == (1, 1, encoder.num_features_per_node)


class TestBatchedGraphs:
    """Batched encoding (all graphs must have same number of nodes)."""

    def test_batch_output_shape(self):
        batch_size, num_nodes = 4, 12
        batch = _make_batch(batch_size, num_nodes)
        encoder = GraphWaveletEncoder(scales=(1, 2, 4), device=torch.device("cpu"))
        out = encoder.encode(batch)
        assert out.shape == (batch_size, num_nodes, encoder.num_features_per_node)

    def test_batch_vs_single_consistent(self):
        """Encoding one graph vs batch-of-one must give identical results."""
        data = _make_graph(10, seed=99)
        batch = Batch.from_data_list([_make_graph(10, seed=99)])
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        out_single = encoder.encode(data)
        out_batch = encoder.encode(batch)
        torch.testing.assert_close(out_single, out_batch)

    def test_batch_deterministic(self):
        batch = _make_batch(3, 8, seed=7)
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        out1 = encoder.encode(batch)
        out2 = encoder.encode(batch)
        torch.testing.assert_close(out1, out2)


class TestScales:
    """Different scale configurations."""

    def test_single_scale(self):
        """With S=1 there are 0 second-order pairs, so output has 1+1=2 features."""
        data = _make_graph(5)
        encoder = GraphWaveletEncoder(scales=(1,), device=torch.device("cpu"))
        out = encoder.encode(data)
        assert encoder.num_features_per_node == 2
        assert out.shape == (1, 5, 2)

    def test_five_scales_default_like_fmri(self):
        """Typical fMRI-style scales (1,2,4,8,16) — 1+5+10 = 16 features."""
        data = _make_graph(100, num_edges=300)
        encoder = GraphWaveletEncoder(scales=(1, 2, 4, 8, 16), device=torch.device("cpu"))
        out = encoder.encode(data)
        assert encoder.num_features_per_node == 16
        assert out.shape == (1, 100, 16)


class TestDevice:
    """Device placement."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_encode_on_cuda(self):
        data = _make_graph(20)
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cuda"))
        out = encoder.encode(data)
        assert out.device.type == "cuda"


class TestForwardAndCall:
    """forward() and __call__() must behave identically to encode()."""

    def test_call_matches_encode(self):
        data = _make_graph(15, seed=77)
        encoder = GraphWaveletEncoder(scales=(1, 2, 4), device=torch.device("cpu"))
        out_encode = encoder.encode(data)
        out_call = encoder(data)
        torch.testing.assert_close(out_call, out_encode)

    def test_forward_matches_encode(self):
        data = _make_graph(15, seed=77)
        encoder = GraphWaveletEncoder(scales=(1, 2, 4), device=torch.device("cpu"))
        out_encode = encoder.encode(data)
        out_forward = encoder.forward(data)
        torch.testing.assert_close(out_forward, out_encode)

    def test_call_with_batch(self):
        batch = _make_batch(3, 10, seed=42)
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        out_encode = encoder.encode(batch)
        out_call = encoder(batch)
        torch.testing.assert_close(out_call, out_encode)


class TestToDevice:
    """.to(device) updates self.device and returns self."""

    def test_to_returns_self(self):
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        result = encoder.to("cpu")
        assert result is encoder

    def test_to_updates_device_string(self):
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        encoder.to("cpu")
        assert encoder.device == torch.device("cpu")

    def test_to_updates_device_torch_device(self):
        encoder = GraphWaveletEncoder(scales=(1,), device=torch.device("cpu"))
        encoder.to(torch.device("cpu"))
        assert encoder.device == torch.device("cpu")

    def test_to_none_is_noop(self):
        encoder = GraphWaveletEncoder(scales=(1, 2), device=torch.device("cpu"))
        result = encoder.to(None)
        assert result is encoder
        assert encoder.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda_and_encode(self):
        encoder = GraphWaveletEncoder(scales=(1, 2)).to("cuda")
        assert encoder.device == torch.device("cuda")
        data = _make_graph(10)
        out = encoder(data)
        assert out.device.type == "cuda"

    def test_chaining(self):
        encoder = GraphWaveletEncoder(scales=(1, 2, 4)).to("cpu")
        assert encoder.device == torch.device("cpu")
        data = _make_graph(8)
        out = encoder(data)
        assert out.shape == (1, 8, encoder.num_features_per_node)


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = [
    # (num_nodes, batch_size, scales)
    # --- vary graph size (batch=1, default scales) ---
    (100,    1, (1, 2, 4, 8, 16)),
    (500,    1, (1, 2, 4, 8, 16)),
    (1000,   1, (1, 2, 4, 8, 16)),
    (2000,   1, (1, 2, 4, 8, 16)),
    (5000,   1, (1, 2, 4, 8, 16)),
    (10000,  1, (1, 2, 4, 8, 16)),
    (20000,  1, (1, 2, 4, 8, 16)),
    (50000,  1, (1, 2, 4, 8, 16)),
    (100000, 1, (1, 2, 4, 8, 16)),
    # --- vary batch size (fixed graph size) ---
    (2000,   2, (1, 2, 4, 8, 16)),
    (2000,   4, (1, 2, 4, 8, 16)),
    # --- vary scale config (fixed graph size) ---
    (2000,   1, (1, 2)),
    (2000,   1, (1, 2, 4, 8)),
]


def _get_rss_mb():
    """Current process RSS in MB (includes PyTorch's C++ allocator)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _benchmark_encode(encoder, data, device, num_runs: int = 3, warmup: int = 1):
    """Benchmark a single config: return (mean_seconds, rss_delta_mb, peak_gpu_mb).

    rss_delta_mb — RSS increase (MB) across the timed runs (process-level).
    peak_gpu_mb  — peak GPU memory delta (MB), or None when on CPU.
    """
    use_cuda = device.type == "cuda"

    for _ in range(warmup):
        encoder.encode(data)
    if use_cuda:
        torch.cuda.synchronize()

    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gpu_before = torch.cuda.memory_allocated(device)

    rss_before = _get_rss_mb()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        encoder.encode(data)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rss_after = _get_rss_mb()
    rss_delta_mb = max(0.0, rss_after - rss_before)

    peak_gpu_mb = None
    if use_cuda:
        gpu_peak = torch.cuda.max_memory_allocated(device)
        peak_gpu_mb = (gpu_peak - gpu_before) / (1024 ** 2)

    return elapsed / num_runs, rss_delta_mb, peak_gpu_mb


def _run_throughput_sweep(device):
    """Run the benchmark sweep on a single device and return formatted lines."""
    label = str(device).upper()
    has_gpu = device.type == "cuda"

    gpu_cols = f"  {'GPU MB':>8}" if has_gpu else ""
    header = (
        f"{'nodes':>7} {'batch':>6} {'scales':>18}"
        f"  {'time (s)':>10} {'ms/graph':>10} {'graphs/s':>10}"
        f"  {'RAM MB':>8}{gpu_cols}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        f"GraphWaveletEncoder throughput — {label}  (mean of 3 runs)",
        sep,
        header,
        sep,
    ]

    for num_nodes, batch_size, scales in BENCHMARK_CONFIGS:
        data = _make_batch(batch_size, num_nodes)
        encoder = GraphWaveletEncoder(scales=scales, device=device)
        mean_s, ram_mb, gpu_mb = _benchmark_encode(encoder, data, device)
        ms_per_graph = mean_s * 1000 / batch_size
        graphs_per_sec = batch_size / mean_s
        scales_str = ",".join(str(s) for s in scales)
        gpu_str = f"  {gpu_mb:>8.1f}" if gpu_mb is not None else ""
        lines.append(
            f"{num_nodes:>7} {batch_size:>6} {scales_str:>18}"
            f"  {mean_s:>10.4f} {ms_per_graph:>10.1f} {graphs_per_sec:>10.2f}"
            f"  {ram_mb:>8.1f}{gpu_str}"
        )

    lines.append(sep)
    return lines


def test_throughput_cpu():
    """Throughput + memory sweep on CPU."""
    lines = _run_throughput_sweep(torch.device("cpu"))
    print("\n".join(lines))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_throughput_gpu():
    """Throughput + memory sweep on GPU."""
    lines = _run_throughput_sweep(torch.device("cuda"))
    print("\n".join(lines))
