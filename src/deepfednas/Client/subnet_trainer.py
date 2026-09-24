from deepfednas.Client.client_trainer import ClientTrainer
import numpy as np
import torch
import logging
from torch import nn
import torch.nn.functional as F


def _mixup_batch(x, labels, alpha, device):
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.size(0), device=device)
    mixed = lam * x + (1.0 - lam) * x[permutation]
    return mixed, labels, labels[permutation], lam


def _cutmix_batch(x, labels, alpha, device):
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.size(0), device=device)
    _, _, height, width = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_height = int(height * cut_ratio)
    cut_width = int(width * cut_ratio)
    center_x = np.random.randint(width)
    center_y = np.random.randint(height)
    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, labels, labels[permutation], lam


def _mix_batch(x, labels, mode, mixup_alpha, cutmix_alpha, device):
    if mode == "alternating":
        mode = "mixup" if np.random.rand() < 0.5 else "cutmix"
    if mode == "mixup":
        return _mixup_batch(x, labels, mixup_alpha, device)
    if mode == "cutmix":
        return _cutmix_batch(x, labels, cutmix_alpha, device)
    if mode == "none":
        return x, labels, labels, 1.0
    raise ValueError(f"Unsupported MixAug mode: {mode!r}")


def _mixed_cross_entropy(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


class SubnetTrainer(ClientTrainer):
    def __init__(self, model, device, args, teacher_model=None):
        super(SubnetTrainer, self).__init__(model, device, args, teacher_model)
        self.test_model = model
        self.alpha = args.feddyn_alpha
        self._mixaug_logged = False

    def set_alpha(self, alpha):
        self.alpha = alpha

    def train(self, lr, local_ep, **kwargs):
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

        self.client_model.to(self.device)
        self.client_model.train()
        if not self.args.use_bn:
            for m in self.client_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    m.running_mean.requires_grad = False
                    m.running_var.requires_grad = False
                    with torch.no_grad():
                        m.weight.fill_(1)
                        m.bias.fill_(0)
                        m.running_mean.fill_(0)
                        m.running_var.fill_(1)

        # train and update
        criterion = nn.CrossEntropyLoss().to(self.device)
        cur_wd = self.args.wd
        if (
            self.client_model.is_max_net(self.client_model.model_config)
            and self.args.largest_subnet_wd
        ):
            cur_wd = self.args.largest_subnet_wd

        if self.args.mod_wd_dyn:
            cur_wd += self.alpha
        model_params = filter(lambda p: p.requires_grad, self.client_model.parameters())

        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=cur_wd,)
        else:
            optimizer = torch.optim.Adam(
                model_params, lr=lr, weight_decay=cur_wd, amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(local_ep if local_ep is not None else self.args.epochs):
            batch_loss = []
            mix_mode = getattr(self.args, "mix_aug_mode", "none")
            mixup_alpha = getattr(self.args, "mixup_alpha", 0.4)
            cutmix_alpha = getattr(self.args, "cutmix_alpha", 1.0)
            if mix_mode != "none" and not self._mixaug_logged:
                logging.info(
                    "MixAug active: mode=%s, mixup_alpha=%s, cutmix_alpha=%s",
                    mix_mode, mixup_alpha, cutmix_alpha,
                )
                self._mixaug_logged = True
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                if mix_mode == "none":
                    labels_a, labels_b, lam = labels, labels, 1.0
                else:
                    x, labels_a, labels_b, lam = _mix_batch(
                        x, labels, mix_mode, mixup_alpha, cutmix_alpha,
                        self.device,
                    )
                self.client_model.zero_grad()
                log_probs = self.client_model.forward(x)
                if self.args.model == "darts":
                    log_probs = log_probs[0]
                if self.args.kd_ratio > 0:
                    with torch.no_grad():
                        soft_logits = self.teacher_model.forward(x).detach()
                        soft_label = F.softmax(soft_logits, dim=1)
                if self.args.kd_ratio == 0:
                    if mix_mode == "none":
                        loss = criterion(log_probs, labels)
                    else:
                        loss = _mixed_cross_entropy(
                            criterion, log_probs, labels_a, labels_b, lam,
                        )
                else:
                    if self.args.kd_type == "ce":
                        kd_loss = self.cross_entropy_loss_with_soft_target(
                            log_probs, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(log_probs, soft_logits)
                    loss = self.args.kd_ratio * kd_loss + (
                        1 - self.args.kd_ratio
                    ) * (
                        criterion(log_probs, labels)
                        if mix_mode == "none"
                        else _mixed_cross_entropy(
                            criterion, log_probs, labels_a, labels_b, lam,
                        )
                    )
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(
                    self.client_model.parameters(), self.args.max_norm
                )

                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if self.args.verbose:
                logging.info(
                    "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                        self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss),
                    )
                )
        if not self.args.use_bn:
            for m in self.client_model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    with torch.no_grad():
                        assert m.weight.equal(torch.ones_like(m.weight)), "BN weight param not all 1s"
                        assert m.bias.equal(torch.zeros_like(m.bias)), "BN bias param not all 0s"
                        assert m.running_mean.equal(torch.zeros_like(m.running_mean)), "BN running mean param not all 0s"
                        assert m.running_var.equal(torch.ones_like(m.running_var)), "BN running var param not all 1s"
        return self.client_model

    def test(self, dataset, args, **kwargs):
        model = self.test_model

        model.to(self.device)
        model.eval()
        if not args.use_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    with torch.no_grad():
                        assert m.weight.equal(torch.ones_like(m.weight)), "BN weight param not all 1s"
                        assert m.bias.equal(torch.zeros_like(m.bias)), "BN bias param not all 0s"
                        assert m.running_mean.equal(torch.zeros_like(m.running_mean)), "BN running mean param not all 0s"
                        assert m.running_var.equal(torch.ones_like(m.running_var)), "BN running var param not all 1s"

        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
            for batch_idx, (x, target) in enumerate(dataset):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model.forward(x)
                if self.args.model == "darts":
                    pred = pred[0]
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
