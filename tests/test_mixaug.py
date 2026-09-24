from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from deepfednas.data.cifar10.data_loader import (
    Cutout as CIFAR10Cutout,
    _data_transforms_cifar10,
)
from deepfednas.data.cifar100.data_loader import (
    Cutout as CIFAR100Cutout,
    _data_transforms_cifar100,
)
from deepfednas.data.cinic10.data_loader import (
    _data_transforms_cinic10,
)
from deepfednas.Client.client_model import (
    ClientModel,
)
from deepfednas.Client import subnet_trainer
from deepfednas.Client.subnet_trainer import (
    SubnetTrainer,
    _cutmix_batch,
    _mixed_cross_entropy,
    _mixup_batch,
)


@pytest.mark.parametrize(
    ("transform_factory", "cutout_type"),
    [
        (_data_transforms_cifar10, CIFAR10Cutout),
        (_data_transforms_cifar100, CIFAR100Cutout),
    ],
)
def test_cifar_mixaug_transform_replaces_cutout_with_randaugment(
    transform_factory, cutout_type,
):
    basic, basic_validation = transform_factory("basic")
    mixaug, mixaug_validation = transform_factory("mixaug")

    assert any(isinstance(transform, cutout_type) for transform in basic.transforms)
    assert not any(
        isinstance(transform, transforms.RandAugment)
        for transform in basic.transforms
    )
    assert any(isinstance(transform, transforms.RandAugment) for transform in mixaug.transforms)
    assert not any(isinstance(transform, cutout_type) for transform in mixaug.transforms)
    assert [type(item) for item in basic_validation.transforms] == [
        type(item) for item in mixaug_validation.transforms
    ]

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert mixaug(image).shape == (3, 32, 32)


def test_cinic10_mixaug_adds_randaugment_without_changing_validation():
    basic, basic_validation = _data_transforms_cinic10("basic")
    mixaug, mixaug_validation = _data_transforms_cinic10("mixaug")

    assert not any(
        isinstance(transform, transforms.RandAugment)
        for transform in basic.transforms
    )
    assert any(
        isinstance(transform, transforms.RandAugment)
        for transform in mixaug.transforms
    )
    assert [type(item) for item in basic_validation.transforms] == [
        type(item) for item in mixaug_validation.transforms
    ]

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert mixaug(image).shape == (3, 32, 32)


@pytest.mark.parametrize(
    "transform_factory",
    [
        _data_transforms_cifar10,
        _data_transforms_cifar100,
        _data_transforms_cinic10,
    ],
)
def test_mixaug_transform_uses_configured_randaugment_policy(transform_factory):
    mixaug, _ = transform_factory(
        "mixaug", randaugment_num_ops=3, randaugment_magnitude=11,
    )
    randaugment = next(
        transform
        for transform in mixaug.transforms
        if isinstance(transform, transforms.RandAugment)
    )

    assert randaugment.num_ops == 3
    assert randaugment.magnitude == 11


def test_mixup_and_mixed_cross_entropy(monkeypatch):
    x = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 3])
    permutation = torch.tensor([3, 2, 1, 0])
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.25)
    monkeypatch.setattr(torch, "randperm", lambda *_args, **_kwargs: permutation)

    mixed, labels_a, labels_b, lam = _mixup_batch(x, labels, 0.4, torch.device("cpu"))
    assert lam == pytest.approx(0.25)
    assert torch.equal(labels_a, labels)
    assert torch.equal(labels_b, labels[permutation])
    assert torch.allclose(mixed, 0.25 * x + 0.75 * x[permutation])

    logits = torch.tensor([
        [3.0, 1.0, 0.0, -1.0],
        [0.0, 3.0, 1.0, -1.0],
        [-1.0, 1.0, 3.0, 0.0],
        [-1.0, 0.0, 1.0, 3.0],
    ])
    criterion = nn.CrossEntropyLoss()
    expected = 0.25 * criterion(logits, labels) + 0.75 * criterion(logits, labels_b)
    assert torch.allclose(
        _mixed_cross_entropy(criterion, logits, labels, labels_b, 0.25),
        expected,
    )


def test_cutmix_replaces_a_nonempty_region(monkeypatch):
    x = torch.stack([
        torch.zeros((3, 8, 8)),
        torch.ones((3, 8, 8)),
    ])
    labels = torch.tensor([0, 1])
    permutation = torch.tensor([1, 0])
    centers = iter((4, 4))
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.5)
    monkeypatch.setattr(np.random, "randint", lambda *_: next(centers))
    monkeypatch.setattr(torch, "randperm", lambda *_args, **_kwargs: permutation)

    mixed, labels_a, labels_b, lam = _cutmix_batch(
        x, labels, 1.0, torch.device("cpu"),
    )
    assert torch.equal(labels_a, labels)
    assert torch.equal(labels_b, labels[permutation])
    assert 0.0 < lam < 1.0
    assert torch.count_nonzero(mixed[0]).item() > 0
    assert torch.count_nonzero(mixed[1] != 1.0).item() > 0


def test_single_subnet_training_calls_mixaug(monkeypatch):
    calls = []

    def record_mix(x, labels, mode, mixup_alpha, cutmix_alpha, device):
        calls.append((mode, mixup_alpha, cutmix_alpha, device.type))
        return x, labels, labels.flip(0), 0.5

    monkeypatch.setattr(subnet_trainer, "_mix_batch", record_mix)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 4))
    client_model = ClientModel(
        model=model,
        model_index=None,
        model_config={"test": True},
        is_max_net=lambda _config: False,
    )
    args = SimpleNamespace(
        feddyn_alpha=0.0,
        use_bn=False,
        wd=5e-4,
        largest_subnet_wd=0.0,
        mod_wd_dyn=False,
        client_optimizer="sgd",
        dataset="cifar100",
        epochs=1,
        model="ofaresnet_generic",
        kd_ratio=0.0,
        kd_type="ce",
        max_norm=10.0,
        verbose=False,
        mix_aug_mode="alternating",
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
    )
    trainer = SubnetTrainer(client_model, torch.device("cpu"), args)
    dataset = TensorDataset(torch.randn(8, 3, 4, 4), torch.arange(8) % 4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    trainer.update_local_dataset(0, loader, loader, len(dataset))
    trainer.train(lr=0.01, local_ep=1)

    assert calls == [
        ("alternating", 0.4, 1.0, "cpu"),
        ("alternating", 0.4, 1.0, "cpu"),
    ]
