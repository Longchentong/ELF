"""Tests for the geometry router and geometry-routed attention.

Run with either:
    PYTHONPATH=src python -m unittest tests.test_geometry_router -v
    PYTHONPATH=src pytest -q tests/test_geometry_router.py   (if pytest installed)
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from modules.geometry_router import (  # noqa: E402
    GeometryRouter, estimate_delta_rel, estimate_spherical_fit,
    masked_token_subsample, pairwise_dist, parse_float_list, parse_layer_spec,
)
from modules.layers import Attention, GeometryRoutedAttention  # noqa: E402
from modules.model import ELF_models  # noqa: E402


class TestParsers(unittest.TestCase):
    def test_parse_float_list(self):
        self.assertEqual(parse_float_list("0.25,0.5,1.0"), [0.25, 0.5, 1.0])

    def test_parse_layer_spec(self):
        self.assertEqual(parse_layer_spec("all", 4), {0, 1, 2, 3})
        self.assertEqual(parse_layer_spec("0,1,2", 12), {0, 1, 2})
        self.assertEqual(parse_layer_spec("0-3,6,8-11", 12),
                         {0, 1, 2, 3, 6, 8, 9, 10, 11})
        with self.assertRaises(ValueError):
            parse_layer_spec("0-13", 12)


class TestRouterShape(unittest.TestCase):
    """Test 1: router gates shape / normalization / finiteness."""

    def test_gates(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 16, 32)
        mask = torch.ones(2, 16)
        t = torch.tensor([0.1, 0.9])
        router = GeometryRouter()
        gates, scores = router(hidden, t, mask)
        self.assertEqual(gates.shape, (2, 3))
        self.assertTrue(torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(gates).all())
        for key in ("e_H", "e_S", "logits", "gates"):
            self.assertTrue(torch.isfinite(scores[key]).all(), key)

    def test_requires_t(self):
        router = GeometryRouter()
        with self.assertRaises(ValueError):
            router(torch.randn(2, 16, 32), None, None)


class TestDeltaRel(unittest.TestCase):
    """Test 2: points on a line are 0-hyperbolic -> small delta_rel."""

    def test_line_is_thin(self):
        pts = torch.arange(16, dtype=torch.float32).view(1, 16, 1) \
            * torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones(1, 16)
        D = pairwise_dist(pts, mask)
        delta_rel = estimate_delta_rel(D, mask, quad_samples=512)
        self.assertLess(delta_rel.item(), 0.1)

    def test_neutral_when_too_few_tokens(self):
        pts = torch.randn(1, 16, 4)
        mask = torch.zeros(1, 16)
        mask[0, :3] = 1  # only 3 valid tokens -> no valid quadruple
        D = pairwise_dist(pts, mask)
        delta_rel = estimate_delta_rel(D, mask, quad_samples=512)
        self.assertEqual(delta_rel.item(), 1.0)


class TestSphericalFit(unittest.TestCase):
    """Test 3: spherical fit is finite and in a sane range."""

    def test_random_no_crash(self):
        torch.manual_seed(1)
        pts = torch.randn(3, 16, 32)
        mask = torch.ones(3, 16)
        D = pairwise_dist(pts, mask)
        e_s = estimate_spherical_fit(D, mask, [0.25, 0.5, 1.0, 2.0, 4.0], rank_dim=32)
        self.assertTrue(torch.isfinite(e_s).all())
        self.assertTrue(((e_s >= 0) & (e_s <= 1)).all())

    def test_rank_penalty_fires(self):
        torch.manual_seed(1)
        pts = torch.randn(1, 16, 32)
        mask = torch.ones(1, 16)
        D = pairwise_dist(pts, mask)
        # rank cap of 2 cannot hold 16 random 32-d points -> nonzero residual.
        e_s = estimate_spherical_fit(D, mask, [0.25, 0.5, 1.0, 2.0, 4.0], rank_dim=1)
        self.assertGreater(e_s.item(), 0.0)

    def test_masked_tokens_ignored(self):
        torch.manual_seed(2)
        base = torch.randn(1, 16, 8)
        mask = torch.ones(1, 16)
        mask[0, 12:] = 0
        poisoned = base.clone()
        poisoned[0, 12:] = 1e3  # garbage in masked positions must not matter
        out = []
        for pts in (base, poisoned):
            s, sm = masked_token_subsample(pts, mask, 16)
            D = pairwise_dist(s, sm)
            out.append(estimate_spherical_fit(D, sm, [0.5, 1.0], rank_dim=8))
        self.assertTrue(torch.allclose(out[0], out[1]))


class TestGeometryRoutedAttention(unittest.TestCase):
    """Test 4: routed attention output shape / finiteness."""

    def test_forward(self):
        torch.manual_seed(0)
        B, N, C, H = 2, 12, 64, 4
        x = torch.randn(B, N, C)
        t = torch.tensor([0.2, 0.8])
        mask = torch.ones(B, N)
        for hyp in ("busemann_proxy", "poincare_distance"):
            for sph in ("cosine", "negative_angular"):
                attn = GeometryRoutedAttention(
                    C, H, geometry_router=GeometryRouter(),
                    hyperbolic_score=hyp, sphere_score=sph)
                out = attn(x, None, attention_mask=mask, t=t, geometry_mask=mask)
                self.assertEqual(out.shape, (B, N, C))
                self.assertTrue(torch.isfinite(out).all(), (hyp, sph))

    def test_fallback_matches_attention(self):
        torch.manual_seed(0)
        B, N, C, H = 2, 12, 64, 4
        x = torch.randn(B, N, C)
        mask = torch.ones(B, N)
        base = Attention(C, H)
        routed = GeometryRoutedAttention(C, H, geometry_router=None)
        routed.load_state_dict(base.state_dict())
        self.assertTrue(torch.allclose(
            base(x, None, attention_mask=mask),
            routed(x, None, attention_mask=mask, t=torch.zeros(B)),
            atol=1e-6,
        ))


class TestDisabledModelParity(unittest.TestCase):
    """Test 5: disabled geometry router keeps the original state_dict + forward."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=False,
        )

    def test_no_geometry_keys_in_state_dict(self):
        geo_keys = [k for k in self.model.state_dict() if "geometry" in k.lower()]
        self.assertEqual(geo_keys, [])
        for block in self.model.blocks:
            self.assertIsInstance(block.attn, Attention)

    def test_forward_smoke(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, 512)
        t = torch.rand(2)
        # self_cond_cfg_scale is required whenever num_self_cond_cfg_tokens > 0
        # (the RoPE buffer reserves slots for those prefix tokens).
        sc = torch.ones(2)
        with torch.no_grad():
            out, dec = self.model(x, t, self_cond_cfg_scale=sc)
        self.assertEqual(out.shape, (2, 16, 512))
        self.assertIsNone(dec)
        self.assertTrue(torch.isfinite(out).all())

    def test_enabled_state_dict_unchanged(self):
        # The router is parameter-free, so even the ENABLED model must keep
        # the exact same state_dict keys (pretrained ckpts stay loadable).
        torch.manual_seed(0)
        routed = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=True, geometry_router_layers="0-3",
        )
        self.assertEqual(set(routed.state_dict()), set(self.model.state_dict()))
        self.assertIsInstance(routed.blocks[0].attn, GeometryRoutedAttention)
        self.assertIsInstance(routed.blocks[4].attn, Attention)


class TestEnabledModelForward(unittest.TestCase):
    """Enabled-router end-to-end forward, with mask and self-conditioning."""

    def test_forward(self):
        torch.manual_seed(0)
        model = ELF_models["ELF-B"](
            text_encoder_dim=512, max_length=16, vocab_size=128,
            num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            geometry_router_enabled=True, geometry_router_layers="0,1",
            geometry_router_sample_size=8, geometry_router_quad_samples=64,
        )
        x = torch.randn(2, 16, 512)
        t = torch.rand(2).clamp(0.05, 0.95)
        mask = torch.ones(2, 16)
        mask[1, 10:] = 0
        sc = torch.full((2,), 1.5)
        with torch.no_grad():
            out, dec = model(x, t, attention_mask=mask,
                             self_cond_cfg_scale=sc, decoder_step_active=True)
        self.assertEqual(out.shape, (2, 16, 512))
        self.assertEqual(dec.shape, (2, 16, 128))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(dec).all())


if __name__ == "__main__":
    unittest.main()
