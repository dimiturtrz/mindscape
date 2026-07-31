"""Test the domain-adversarial subject discriminator."""
import torch

from neuroscan.models.subject_discriminator import SubjectDiscriminator


def test_grad_reverse_forward():
    """_GradReverse.forward is the IDENTITY (it only touches the backward pass) — so the discriminator's forward
    value is exactly its net applied to the input, unaffected by λ."""
    torch.manual_seed(0)
    disc = SubjectDiscriminator(embed_dim=32, n_subjects=4, hidden=16)
    z = torch.randn(8, 32)
    assert torch.equal(disc.forward(z, 2.0), disc.net(z))        # GRL forward is a pass-through


def test_subject_discriminator_forward():
    """SubjectDiscriminator.forward maps [n, embed] -> [n, n_subjects] class logits."""
    torch.manual_seed(0)
    n_subjects, embed_dim = 5, 64
    disc = SubjectDiscriminator(embed_dim=embed_dim, n_subjects=n_subjects, hidden=128)
    z = torch.randn(16, embed_dim, requires_grad=True)
    out = disc.forward(z, 1.0)
    assert out.shape == (16, n_subjects)
    assert out.dtype == torch.float32


def test_backward():
    """_GradReverse.backward SIGN-FLIPS the gradient and scales it by λ — the whole point of the GRL: the
    upstream sees `-λ · g`, so minimizing the discriminator loss pushes the encoder the OTHER way."""
    z = torch.randn(8, 32, requires_grad=True)
    lambd = 2.0
    SubjectDiscriminator._GradReverse.apply(z, lambd).sum().backward()   # d(sum)/dz = 1 before reversal
    assert torch.allclose(z.grad, torch.full_like(z, -lambd))            # reversed + scaled: grad = -λ·1
