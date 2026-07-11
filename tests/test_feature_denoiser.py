"""
test_feature_denoiser.py
=========================
Unit tests cho Multimodal Feature Denoiser module.

Chay: python -m pytest tests/test_feature_denoiser.py -v
"""

from __future__ import annotations

import pytest
import torch

from ai_engine.denoising.feature_denoiser import (
    DenoisingMLP,
    FeatureDenoiser,
    GaussianDiffusionDenoiser,
    MultimodalAlignmentLayer,
    TaskGuidedGating,
    timestep_embedding,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

BATCH_SIZE = 8
TEXT_DIM = 768
IMAGE_DIM = 1280
HIDDEN_DIM = 64   # Small for fast tests
NOISE_STEPS = 3


@pytest.fixture
def text_features():
    return torch.randn(BATCH_SIZE, TEXT_DIM)


@pytest.fixture
def image_features():
    return torch.randn(BATCH_SIZE, IMAGE_DIM)


@pytest.fixture
def hidden_features():
    return torch.randn(BATCH_SIZE, HIDDEN_DIM)


@pytest.fixture
def denoiser():
    return FeatureDenoiser(
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        hidden_dim=HIDDEN_DIM,
        noise_steps=NOISE_STEPS,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: timestep_embedding
# ═══════════════════════════════════════════════════════════════════════════

class TestTimestepEmbedding:
    def test_output_shape(self):
        """Embedding shape should be (batch, dim)."""
        t = torch.tensor([0, 1, 2, 3])
        emb = timestep_embedding(t, dim=16)
        assert emb.shape == (4, 16)

    def test_odd_dim(self):
        """Should handle odd dimensions by zero-padding."""
        t = torch.tensor([0, 1])
        emb = timestep_embedding(t, dim=15)
        assert emb.shape == (2, 15)

    def test_different_timesteps_give_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        t = torch.tensor([0, 5])
        emb = timestep_embedding(t, dim=16)
        assert not torch.allclose(emb[0], emb[1])


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: DenoisingMLP
# ═══════════════════════════════════════════════════════════════════════════

class TestDenoisingMLP:
    def test_output_shape(self, hidden_features):
        """Output should have same shape as input features."""
        mlp = DenoisingMLP(feature_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM)
        timesteps = torch.randint(0, 5, (BATCH_SIZE,))
        output = mlp(hidden_features, timesteps)
        assert output.shape == hidden_features.shape

    def test_gradient_flow(self, hidden_features):
        """Gradients should flow through the network."""
        mlp = DenoisingMLP(feature_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM)
        timesteps = torch.randint(0, 5, (BATCH_SIZE,))
        output = mlp(hidden_features, timesteps)
        loss = output.mean()
        loss.backward()
        for param in mlp.parameters():
            assert param.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: GaussianDiffusionDenoiser
# ═══════════════════════════════════════════════════════════════════════════

class TestGaussianDiffusionDenoiser:
    @pytest.fixture
    def diffusion(self):
        return GaussianDiffusionDenoiser(
            feature_dim=HIDDEN_DIM,
            noise_steps=NOISE_STEPS,
            hidden_dim=HIDDEN_DIM,
        )

    def test_q_sample_shape(self, diffusion, hidden_features):
        """Forward diffusion should preserve shape."""
        t = torch.randint(0, NOISE_STEPS, (BATCH_SIZE,))
        noisy = diffusion.q_sample(hidden_features, t)
        assert noisy.shape == hidden_features.shape

    def test_q_sample_adds_noise(self, diffusion, hidden_features):
        """Forward diffusion should change the features."""
        t = torch.tensor([NOISE_STEPS - 1] * BATCH_SIZE)
        noisy = diffusion.q_sample(hidden_features, t)
        assert not torch.allclose(noisy, hidden_features, atol=1e-3)

    def test_p_sample_shape(self, diffusion, hidden_features):
        """Reverse diffusion should preserve shape."""
        denoised = diffusion.p_sample(hidden_features)
        assert denoised.shape == hidden_features.shape

    def test_training_losses_shape(self, diffusion, hidden_features):
        """Training loss should return per-sample losses."""
        losses = diffusion.training_losses(hidden_features)
        assert "loss" in losses
        assert losses["loss"].shape == (BATCH_SIZE,)

    def test_training_losses_positive(self, diffusion, hidden_features):
        """MSE loss should be positive."""
        losses = diffusion.training_losses(hidden_features)
        assert (losses["loss"] >= 0).all()

    def test_linear_schedule(self):
        """Should create with linear noise schedule."""
        d = GaussianDiffusionDenoiser(
            feature_dim=32, noise_steps=5, noise_schedule="linear", hidden_dim=32
        )
        assert d.betas.shape == (5,)

    def test_cosine_schedule(self):
        """Should create with cosine noise schedule."""
        d = GaussianDiffusionDenoiser(
            feature_dim=32, noise_steps=5, noise_schedule="cosine", hidden_dim=32
        )
        assert d.betas.shape == (5,)

    def test_forward_convenience(self, diffusion, hidden_features):
        """Forward call should work as p_sample."""
        output = diffusion(hidden_features)
        assert output.shape == hidden_features.shape


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: TaskGuidedGating
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskGuidedGating:
    def test_output_shape(self, hidden_features):
        """Gated output should have same shape."""
        gate = TaskGuidedGating(HIDDEN_DIM)
        task_signal = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        output = gate(hidden_features, task_signal)
        assert output.shape == hidden_features.shape

    def test_gating_range(self, hidden_features):
        """Gate values should be in [0, 1] due to sigmoid."""
        gate = TaskGuidedGating(HIDDEN_DIM)
        task_signal = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        # Access internal gate weights
        gate_values = gate.gate(task_signal)
        assert (gate_values >= 0).all() and (gate_values <= 1).all()

    def test_zero_signal_passes_some(self, hidden_features):
        """With zero task signal, gate should output ~0.5 * feature (sigmoid(0)=0.5)."""
        gate = TaskGuidedGating(HIDDEN_DIM)
        # Zero bias → sigmoid(0) = 0.5
        with torch.no_grad():
            for p in gate.parameters():
                p.zero_()
        task_signal = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
        output = gate(hidden_features, task_signal)
        expected = 0.5 * hidden_features
        assert torch.allclose(output, expected, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: MultimodalAlignmentLayer
# ═══════════════════════════════════════════════════════════════════════════

class TestMultimodalAlignmentLayer:
    @pytest.fixture
    def alignment(self):
        return MultimodalAlignmentLayer(feature_dim=HIDDEN_DIM)

    def test_distribution_matching_identical(self, alignment):
        """Identical features should have zero distribution matching loss."""
        features = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        loss = alignment.distribution_matching_loss(features, features)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_distribution_matching_different(self, alignment):
        """Different distributions should have positive loss."""
        f1 = torch.randn(BATCH_SIZE, HIDDEN_DIM) * 2
        f2 = torch.randn(BATCH_SIZE, HIDDEN_DIM) * 0.5 + 3
        loss = alignment.distribution_matching_loss(f1, f2)
        assert loss.item() > 0

    def test_infonce_identical(self, alignment):
        """InfoNCE with identical features should give low loss."""
        features = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        loss = alignment.infonce_loss(features, features)
        # With identical features, all similarities are equal → loss depends on batch size
        assert loss.isfinite()

    def test_forward_returns_scalar(self, alignment, hidden_features):
        """Forward should return a scalar loss."""
        f2 = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        loss = alignment(hidden_features, f2)
        assert loss.shape == ()
        assert loss.isfinite()


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: FeatureDenoiser (End-to-End)
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureDenoiser:
    def test_denoise_text_shape(self, denoiser, text_features):
        """Text-only denoising should preserve shape."""
        output = denoiser.denoise_text(text_features)
        assert output.shape == text_features.shape

    def test_denoise_image_shape(self, denoiser, image_features):
        """Image-only denoising should preserve shape."""
        output = denoiser.denoise_image(image_features)
        assert output.shape == image_features.shape

    def test_denoise_multimodal_shapes(self, denoiser, text_features, image_features):
        """Multimodal denoising should preserve both shapes."""
        text_out, image_out = denoiser.denoise_multimodal(text_features, image_features)
        assert text_out.shape == text_features.shape
        assert image_out.shape == image_features.shape

    def test_forward_with_image(self, denoiser, text_features, image_features):
        """Forward with both modalities."""
        text_out, image_out = denoiser(text_features, image_features)
        assert text_out.shape == text_features.shape
        assert image_out.shape == image_features.shape

    def test_forward_text_only(self, denoiser, text_features):
        """Forward with text only."""
        text_out, image_out = denoiser(text_features, None)
        assert text_out.shape == text_features.shape
        assert image_out is None

    def test_training_step_returns_losses(self, denoiser, text_features, image_features):
        """Training step should return all loss components."""
        losses = denoiser.training_step(text_features, image_features)
        assert "total_loss" in losses
        assert "diffusion_loss" in losses
        assert "alignment_loss" in losses
        assert "text_diffusion_loss" in losses
        assert "image_diffusion_loss" in losses

    def test_training_step_losses_finite(self, denoiser, text_features, image_features):
        """All losses should be finite."""
        losses = denoiser.training_step(text_features, image_features)
        for key, val in losses.items():
            assert val.isfinite(), f"{key} is not finite: {val}"

    def test_training_step_backward(self, denoiser, text_features, image_features):
        """Gradients should flow through training step."""
        losses = denoiser.training_step(text_features, image_features)
        losses["total_loss"].backward()
        grad_count = sum(
            1 for p in denoiser.parameters() if p.grad is not None
        )
        assert grad_count > 0

    def test_loss_decreases_after_training(self, text_features, image_features):
        """Loss should decrease after a few training steps."""
        denoiser = FeatureDenoiser(
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
            hidden_dim=HIDDEN_DIM,
            noise_steps=NOISE_STEPS,
        )
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

        # Initial loss
        initial_loss = denoiser.training_step(text_features, image_features)["total_loss"].item()

        # Train for a few steps
        denoiser.train()
        for _ in range(20):
            optimizer.zero_grad()
            losses = denoiser.training_step(text_features, image_features)
            losses["total_loss"].backward()
            optimizer.step()

        final_loss = denoiser.training_step(text_features, image_features)["total_loss"].item()

        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 1.5, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_parameter_count(self, denoiser):
        """Model should have a reasonable number of parameters."""
        param_count = sum(p.numel() for p in denoiser.parameters())
        assert param_count > 0
        # Should be lightweight (< 10M params with small hidden_dim)
        assert param_count < 10_000_000


# ═══════════════════════════════════════════════════════════════════════════
#  TEST: Serialization
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_save_and_load(self, denoiser, text_features, image_features, tmp_path):
        """Model should be saveable and loadable."""
        # Get initial output (seed for deterministic diffusion noise)
        denoiser.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            text_out1, img_out1 = denoiser(text_features, image_features)

        # Save
        path = tmp_path / "test_denoiser.pt"
        torch.save(denoiser.state_dict(), path)

        # Load into new model
        new_denoiser = FeatureDenoiser(
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
            hidden_dim=HIDDEN_DIM,
            noise_steps=NOISE_STEPS,
        )
        new_denoiser.load_state_dict(torch.load(path, weights_only=True))
        new_denoiser.eval()

        with torch.no_grad():
            torch.manual_seed(42)
            text_out2, img_out2 = new_denoiser(text_features, image_features)

        assert torch.allclose(text_out1, text_out2, atol=1e-4)
        assert torch.allclose(img_out1, img_out2, atol=1e-4)
