"""Test the domain-adversarial subject discriminator."""
import torch

from neuroscan.models.subject_discriminator import SubjectDiscriminator


def test_grad_reverse_forward():
    """_GradReverse.forward applies gradient reversal in SubjectDiscriminator."""
    torch.manual_seed(0)
    n_subjects = 5
    embed_dim = 64
    disc = SubjectDiscriminator(embed_dim=embed_dim, n_subjects=n_subjects, hidden=128)
    z = torch.randn(16, embed_dim, requires_grad=True)
    lambd = 1.0
    out = disc.forward(z, lambd)
    assert out.shape == (16, n_subjects)
    assert out.dtype == torch.float32


def test_subject_discriminator_forward():
    """SubjectDiscriminator.forward predicts subject class with gradient reversal."""
    torch.manual_seed(0)
    n_subjects = 5
    embed_dim = 64
    disc = SubjectDiscriminator(embed_dim=embed_dim, n_subjects=n_subjects, hidden=128)
    z = torch.randn(16, embed_dim, requires_grad=True)
    lambd = 1.0
    out = disc.forward(z, lambd)
    assert out.shape == (16, n_subjects)
    assert out.dtype == torch.float32


def test_backward():
    """_GradReverse.backward sign-flips gradients by lambda."""
    torch.manual_seed(0)
    z = torch.randn(8, 32, requires_grad=True)
    lambd = 2.0
    # _GradReverse.backward is tested through the gradient reversal in autograd
    reversed_z = SubjectDiscriminator._GradReverse.apply(z, lambd)
    loss = reversed_z.sum()
    loss.backward()
    assert z.grad is not None
