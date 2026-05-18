import pytest


torch = pytest.importorskip("torch")

from PipelineTS.nn_model.layers import ExpertRouter, FreqMixingBlock, GlobalTemporalBlock


def test_expert_router_uses_sparse_topk_and_load_bias():
    router = ExpertRouter(d_model=8, n_experts=3, top_k=2, noise_std=0.0)
    router.train()
    x = torch.randn(16, 12, 8)

    weights, indices, aux_loss = router(x)

    assert weights.shape == (16, 3)
    assert indices.shape == (16, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(16), atol=1e-6)
    assert torch.all((weights > 0).sum(dim=1) == 2)
    assert aux_loss.ndim == 0
    assert router._load_bias.abs().sum().item() > 0

    stats = router.get_routing_stats()
    assert len(stats["expert_freq"]) == 3
    assert len(stats["load_bias"]) == 3
    assert stats["total_samples"] == 16

    router.reset_stats()
    reset_stats = router.get_routing_stats()
    assert reset_stats["total_samples"] == 0
    assert reset_stats["load_bias"] == [0.0, 0.0, 0.0]


def test_global_temporal_block_adaptive_forward_shape_and_aux_loss():
    block = GlobalTemporalBlock(seq_len=12, d_model=16, routing_mode="adaptive", top_k_experts=2)
    block.train()
    x = torch.randn(8, 12)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert block.get_aux_loss().ndim == 0


def test_expert_router_dynamic_topk_tracks_active_experts():
    router = ExpertRouter(d_model=8, n_experts=6, top_k="dynamic", noise_std=0.0, min_top_k=2, max_top_k=4)
    router.train()
    x = torch.randn(10, 12, 8)

    weights, indices, aux_loss = router(x)
    active = (weights > 0).sum(dim=1)
    stats = router.get_routing_stats()

    assert weights.shape == (10, 6)
    assert indices.shape == (10, 4)
    assert active.min().item() >= 2
    assert active.max().item() <= 4
    assert torch.allclose(weights.sum(dim=1), torch.ones(10), atol=1e-6)
    assert aux_loss.ndim == 0
    assert stats["dynamic_top_k"] is True
    assert 2 <= stats["avg_active_experts"] <= 4


def test_expert_router_feature_adaptive_pool_grows_for_complex_series():
    router = ExpertRouter(
        d_model=4,
        n_experts=7,
        top_k="dynamic",
        noise_std=0.0,
        min_top_k=2,
        max_top_k=4,
        feature_adaptive=True,
        pool_min_experts=4,
        pool_max_experts=7,
    )
    router.train()
    simple = torch.zeros(8, 24, 4)
    t = torch.arange(24, dtype=torch.float32)
    complex_signal = (
        torch.sin(2 * torch.pi * t / 5)
        + 0.5 * torch.sin(2 * torch.pi * t / 3)
        + torch.linspace(0, 4, 24)
    )
    complex_signal = complex_signal.view(1, 24, 1).repeat(8, 1, 4)
    complex_signal[:, 6::7, :] += 6

    router(simple)
    simple_stats = router.get_routing_stats()
    router.reset_stats()
    router(complex_signal)
    complex_stats = router.get_routing_stats()

    assert simple_stats["feature_adaptive"] is True
    assert complex_stats["feature_adaptive"] is True
    assert len(complex_stats["avg_profile"]) == 8
    assert simple_stats["avg_candidate_experts"] == 4.0
    assert complex_stats["avg_candidate_experts"] > simple_stats["avg_candidate_experts"]
    assert complex_stats["avg_active_experts"] >= simple_stats["avg_active_experts"]


def test_global_temporal_block_adaptive_plus_uses_extended_experts():
    block = GlobalTemporalBlock(seq_len=16, d_model=16, routing_mode="adaptive_plus")
    block.train()
    x = torch.randn(6, 16)

    out = block(x)
    stats = block.get_routing_stats()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert stats["dynamic_top_k"] is True
    assert stats["feature_adaptive"] is True
    assert len(stats["expert_names"]) == 7
    assert len(stats["expert_freq"]) == 7
    assert "Trend" in stats["expert_names"]
    assert "SeasonalResidual" in stats["expert_names"]
    assert "NoiseResidual" in stats["expert_names"]
    assert "Skip" in stats["expert_names"]
    assert 4 <= stats["avg_candidate_experts"] <= 7
    assert 2 <= stats["avg_active_experts"] <= 4


def test_freq_mixing_block_weighted_period_fusion_is_finite():
    block = FreqMixingBlock(d_model=4, top_k=3)
    t = torch.arange(24, dtype=torch.float32)
    signal = torch.sin(2 * torch.pi * t / 6).view(1, 24, 1).repeat(3, 1, 4)

    out = block(signal)

    assert out.shape == signal.shape
    assert torch.isfinite(out).all()
