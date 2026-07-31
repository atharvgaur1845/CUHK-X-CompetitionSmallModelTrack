#!/usr/bin/env python3
"""Cross-user supervised-contrastive pretraining for the skeleton MultiTCN.

The outer-fold validation users are never loaded by an optimizer-facing
DataLoader and are evaluated exactly once, after fixed pretrain/fine-tune
epochs.  Saved checkpoints are plain MultiStreamTCN state dicts, matching the
format used by train.py, fuse_submit.py, and tta_probe.py.
"""
import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

import har_data
import har_models


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "research", "artifacts", "results.csv")
ARTIFACTS = os.path.dirname(RESULTS)
CKPT = os.path.join(ROOT, "checkpoints")


def seed_everything(seed):
    """Make initialization, sampling, and CUDA kernels reproducible."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class CrossUserClassBatchSampler(Sampler):
    """Class-balanced batches with same-class samples from multiple users.

    Every selected class contributes ``samples_per_class`` examples.  Its first
    two examples always come from distinct users; remaining examples are drawn
    across the class's users.  Classes with fewer than two training users are
    excluded from contrastive pretraining and are still seen during the
    natural-prior fine-tuning stage.
    """

    def __init__(
        self,
        dataset,
        classes_per_batch=16,
        samples_per_class=4,
        seed=0,
        batches_per_epoch=None,
    ):
        if classes_per_batch < 1:
            raise ValueError("classes_per_batch must be positive")
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be at least 2")
        self.dataset = dataset
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0

        grouped = defaultdict(lambda: defaultdict(list))
        for index, sid in enumerate(dataset.ids):
            grouped[int(dataset.labels[sid])][dataset.meta[sid]["user"]].append(index)
        self.by_class_user = {
            label: {user: tuple(indices) for user, indices in sorted(by_user.items())}
            for label, by_user in sorted(grouped.items())
        }
        self.eligible_classes = tuple(
            label for label, by_user in self.by_class_user.items() if len(by_user) >= 2
        )
        self.ineligible_classes = tuple(
            label for label, by_user in self.by_class_user.items() if len(by_user) < 2
        )
        if len(self.eligible_classes) < self.classes_per_batch:
            raise ValueError(
                f"need at least {self.classes_per_batch} classes represented by >=2 users; "
                f"found {len(self.eligible_classes)}"
            )
        batch_size = self.classes_per_batch * self.samples_per_class
        self.batches_per_epoch = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else max(1, math.ceil(len(dataset) / batch_size))
        )
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")

    @property
    def batch_size(self):
        return self.classes_per_batch * self.samples_per_class

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        usage = {label: 0 for label in self.eligible_classes}
        queues, positions = {}, {}
        for label in self.eligible_classes:
            for user, indices in self.by_class_user[label].items():
                key = (label, user)
                queues[key] = list(indices)
                rng.shuffle(queues[key])
                positions[key] = 0

        def draw_index(label, user, used):
            """Cycle without replacement, avoiding within-class duplicates."""
            key = (label, user)
            queue = queues[key]
            for _ in range(len(queue)):
                if positions[key] == len(queue):
                    rng.shuffle(queue)
                    positions[key] = 0
                index = queue[positions[key]]
                positions[key] += 1
                if index not in used:
                    return index
            # Every clip for this user is already present in the class group.
            # A repeat is then mathematically unavoidable.
            if positions[key] == len(queue):
                rng.shuffle(queue)
                positions[key] = 0
            index = queue[positions[key]]
            positions[key] += 1
            return index

        for _ in range(self.batches_per_epoch):
            # Random tie-breaking among the least-used labels keeps cumulative
            # class counts within one occurrence of each other.
            tie_break = {label: float(rng.random()) for label in self.eligible_classes}
            chosen = sorted(
                self.eligible_classes, key=lambda label: (usage[label], tie_break[label])
            )[: self.classes_per_batch]
            batch = []
            for label in chosen:
                usage[label] += 1
                by_user = self.by_class_user[label]
                users = list(by_user)
                rng.shuffle(users)
                selected_users = users[: min(len(users), self.samples_per_class)]
                selected_count = defaultdict(int)
                for user in selected_users:
                    selected_count[user] += 1
                while len(selected_users) < self.samples_per_class:
                    # Prefer users that still have an unused clip for this class.
                    candidates = [
                        user
                        for user in users
                        if selected_count[user] < len(by_user[user])
                    ]
                    pool = candidates or users
                    user = pool[int(rng.integers(len(pool)))]
                    selected_users.append(user)
                    selected_count[user] += 1
                used = set()
                for user in selected_users:
                    index = draw_index(label, user, used)
                    batch.append(index)
                    used.add(index)
            rng.shuffle(batch)
            yield batch


def cross_user_supcon_loss(features, labels, users, temperature=0.1):
    """Supervised contrastive loss whose positives cross subject identity.

    Same-class examples from the same user remain in the denominator but are
    not positives, explicitly discouraging subject-specific shortcuts.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [batch, dim], got {tuple(features.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(features) != len(labels) or len(features) != len(users):
        raise ValueError("features, labels, and users must have matching lengths")

    # Similarities and logsumexp stay in fp32 even when the encoders use AMP.
    with torch.autocast(device_type=features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=1)
        logits = z @ z.T / float(temperature)
        self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(self_mask, -torch.inf)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        labels = labels.reshape(-1)
        users = users.reshape(-1)
        positives = (
            labels[:, None].eq(labels[None, :])
            & users[:, None].ne(users[None, :])
            & ~self_mask
        )
        positive_count = positives.sum(1)
        valid = positive_count > 0
        if not bool(valid.any()):
            raise ValueError("batch has no same-class, different-user positive pairs")
        positive_log_prob = torch.where(
            positives, log_prob, torch.zeros_like(log_prob)
        ).sum(1)
        return -(positive_log_prob[valid] / positive_count[valid]).mean()


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def optimizer_step(loss, optimizer, scaler, scheduler, parameters, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


def pretrain(args, fold, dataset, model, device, amp_enabled):
    sampler = CrossUserClassBatchSampler(
        dataset,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        seed=args.seed + fold * 10_007,
        batches_per_epoch=args.pretrain_batches or None,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    projector = ProjectionHead(
        model.feature_dim, args.projection_hidden, args.projection_dim
    ).to(device)
    parameters = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        epochs=args.pretrain_epochs,
        steps_per_epoch=len(loader),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    user_to_index = {
        user: index
        for index, user in enumerate(sorted({dataset.meta[sid]["user"] for sid in dataset.ids}))
    }

    for epoch in range(args.pretrain_epochs):
        model.train()
        projector.train()
        sampler.set_epoch(epoch)
        dataset.epoch_seed = (args.seed + fold * 10_007) * 1_000 + epoch
        running_loss = running_supcon = running_ce = 0.0
        valid_anchors = examples = 0
        for x, y, sids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            users = torch.tensor(
                [user_to_index[dataset.meta[sid]["user"]] for sid in sids],
                dtype=torch.long,
                device=device,
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                features = model.forward_features(x)
                embeddings = projector(features)
                supcon_loss = cross_user_supcon_loss(
                    embeddings, y, users, args.temperature
                )
                logits = model.head(model.drop(features))
                ce_loss = ce_criterion(logits, y)
                loss = args.supcon_weight * supcon_loss + args.aux_ce_weight * ce_loss
            optimizer_step(
                loss, optimizer, scaler, scheduler, parameters, args.grad_clip
            )
            with torch.no_grad():
                pos = y[:, None].eq(y[None, :]) & users[:, None].ne(users[None, :])
                valid_anchors += int(pos.any(1).sum().item())
                examples += len(y)
            running_loss += float(loss.detach()) * len(y)
            running_supcon += float(supcon_loss.detach()) * len(y)
            running_ce += float(ce_loss.detach()) * len(y)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.pretrain_epochs:
            print(
                f"  fold{fold} pretrain {epoch + 1}/{args.pretrain_epochs}: "
                f"loss {running_loss / examples:.4f} "
                f"supcon {running_supcon / examples:.4f} "
                f"ce {running_ce / examples:.4f} "
                f"valid anchors {valid_anchors}/{examples}",
                flush=True,
            )
    return sampler


def finetune(args, fold, dataset, model, device, amp_enabled):
    generator = torch.Generator()
    generator.manual_seed(args.seed + fold * 10_007 + 71)
    loader = DataLoader(
        dataset,
        batch_size=args.finetune_bs,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    if not len(loader):
        raise ValueError("fine-tuning DataLoader has zero batches")
    optimizer = torch.optim.AdamW(
        model.parameters(), args.finetune_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.finetune_lr,
        epochs=args.finetune_epochs,
        steps_per_epoch=len(loader),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    parameters = list(model.parameters())

    for epoch in range(args.finetune_epochs):
        model.train()
        dataset.epoch_seed = (
            (args.seed + fold * 10_007) * 1_000 + args.pretrain_epochs + epoch
        )
        running_loss = examples = 0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss = criterion(model(x), y)
            optimizer_step(
                loss, optimizer, scaler, scheduler, parameters, args.grad_clip
            )
            running_loss += float(loss.detach()) * len(y)
            examples += len(y)
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.finetune_epochs:
            print(
                f"  fold{fold} finetune {epoch + 1}/{args.finetune_epochs}: "
                f"loss {running_loss / examples:.4f} (outer validation untouched)",
                flush=True,
            )


def evaluate_once(args, fold, dataset, model, device):
    """One and only outer-validation pass for this training run."""
    loader = DataLoader(
        dataset,
        batch_size=args.eval_bs,
        shuffle=False,
        num_workers=max(0, args.workers // 2),
        pin_memory=device.type == "cuda",
    )
    model.eval()
    probs, labels, sids = [], [], []
    with torch.no_grad():
        for x, y, batch_sids in loader:
            p = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
            sids.extend(batch_sids)
    probabilities = np.concatenate(probs)
    targets = np.concatenate(labels)
    accuracy = float((probabilities.argmax(1) == targets).mean())
    class_accs = [
        float((probabilities[targets == label].argmax(1) == label).mean())
        for label in np.unique(targets)
    ]
    macro = float(np.mean(class_accs))
    print(
        f"  fold{fold} final outer evaluation: micro {accuracy:.4f} macro {macro:.4f}",
        flush=True,
    )
    return accuracy, macro, probabilities, targets, sids


def build_fold_datasets(args, fold):
    train = har_data.HARDataset(
        "train",
        "skel",
        fold,
        "train",
        aug=args.aug_spec,
        seed=args.seed,
        feat="jvb",
    )
    outer_val = har_data.HARDataset(
        "train", "skel", fold, "val", aug=False, feat="jvb"
    )
    train_ids, val_ids = set(train.ids), set(outer_val.ids)
    train_users = {train.meta[sid]["user"] for sid in train.ids}
    val_users = {outer_val.meta[sid]["user"] for sid in outer_val.ids}
    if train_ids & val_ids:
        raise RuntimeError(f"fold {fold}: outer train/validation sample leakage")
    if train_users & val_users:
        raise RuntimeError(f"fold {fold}: outer train/validation user leakage")
    if not train.ids or not outer_val.ids:
        raise RuntimeError(f"fold {fold}: empty outer train or validation partition")
    return train, outer_val, train_users, val_users


def run_fold(args, fold):
    fold_seed = args.seed + fold * 10_007
    seed_everything(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    train, outer_val, train_users, val_users = build_fold_datasets(args, fold)
    model = har_models.build(
        "skel", in_ch=train.skel_dim(), arch="multitcn", width=args.width
    ).to(device)
    if not hasattr(model, "forward_features"):
        raise TypeError("selected model must expose forward_features()")
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"fold {fold}: {len(train)} train clips/{len(train_users)} users, "
        f"{len(outer_val)} outer-val clips/{len(val_users)} held-out users, "
        f"{n_params / 1e6:.2f}M params, device={device}, amp={amp_enabled}",
        flush=True,
    )

    sampler = pretrain(args, fold, train, model, device, amp_enabled)
    if sampler.ineligible_classes:
        print(
            "  contrastive stage excluded classes with <2 training users: "
            + ",".join(map(str, sampler.ineligible_classes)),
            flush=True,
        )
    finetune(args, fold, train, model, device, amp_enabled)
    accuracy, macro, probs, labels, sids = evaluate_once(
        args, fold, outer_val, model, device
    )

    os.makedirs(CKPT, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    # Both files intentionally contain the same fixed-epoch model.  There is no
    # outer-validation checkpoint selection.
    torch.save(state, os.path.join(CKPT, f"{args.tag}_f{fold}.pt"))
    torch.save(state, os.path.join(CKPT, f"{args.tag}_f{fold}_last.pt"))
    return {
        "accuracy": accuracy,
        "macro": macro,
        "probs": probs,
        "labels": labels,
        "sids": sids,
        "params": n_params,
        "train_users": sorted(train_users),
        "outer_val_users": sorted(val_users),
        "ineligible_classes": list(sampler.ineligible_classes),
    }


def save_run_artifacts(args, folds, fold_results, elapsed_minutes):
    accuracies = [result["accuracy"] for result in fold_results]
    macros = [result["macro"] for result in fold_results]
    n_params = fold_results[0]["params"]
    row = {
        "exp": args.exp,
        "tag": args.tag,
        "modality": "skel",
        "aug": int(bool(args.aug_spec)),
        "epochs": args.pretrain_epochs + args.finetune_epochs,
        "seed": args.seed,
        "folds": args.folds,
        "accs": json.dumps([round(value, 4) for value in accuracies]),
        "mean": round(float(np.mean(accuracies)), 4),
        "std": round(float(np.std(accuracies)), 4),
        "params": n_params,
        "mins": round(elapsed_minutes, 1),
    }
    os.makedirs(ARTIFACTS, exist_ok=True)
    new_results_file = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_results_file:
            writer.writeheader()
        writer.writerow(row)

    all_four = folds == [0, 1, 2, 3]
    fold_suffix = "" if all_four else "_f" + "".join(map(str, folds))
    oof_path = os.path.join(ARTIFACTS, f"oof_{args.tag}{fold_suffix}.npz")
    np.savez_compressed(
        oof_path,
        probs=np.concatenate([result["probs"] for result in fold_results]),
        labels=np.concatenate([result["labels"] for result in fold_results]),
        sids=np.asarray(
            [sid for result in fold_results for sid in result["sids"]], dtype=str
        ),
    )

    manifest = {
        "exp": args.exp,
        "tag": args.tag,
        "args": vars(args),
        "selection_protocol": (
            "fixed pretrain/fine-tune epochs; outer validation evaluated once after training"
        ),
        "folds": folds,
        "fold_accuracies": accuracies,
        "fold_macro_accuracies": macros,
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "params": n_params,
        "elapsed_minutes": elapsed_minutes,
        "oof_path": os.path.relpath(oof_path, ROOT),
        "partitions": [
            {
                "fold": fold,
                "train_users": result["train_users"],
                "outer_val_users": result["outer_val_users"],
                "contrastive_ineligible_classes": result["ineligible_classes"],
            }
            for fold, result in zip(folds, fold_results)
        ],
        "checkpoint_config": {
            "modality": "skel",
            "build_kwargs": {"arch": "multitcn", "width": args.width},
            "dataset_kwargs": {"feat": "jvb"},
        },
    }
    safe_folds = args.folds.replace(",", "-")
    manifest_path = os.path.join(
        ARTIFACTS, f"run_{args.exp}_{args.tag}_{safe_folds}.json"
    )
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return row, oof_path, manifest_path


def dry_run(args, folds):
    for fold in folds:
        train, outer_val, train_users, val_users = build_fold_datasets(args, fold)
        sampler = CrossUserClassBatchSampler(
            train,
            classes_per_batch=args.classes_per_batch,
            samples_per_class=args.samples_per_class,
            seed=args.seed + fold * 10_007,
            batches_per_epoch=args.pretrain_batches or None,
        )
        model = har_models.build(
            "skel", in_ch=train.skel_dim(), arch="multitcn", width=args.width
        )
        print(
            f"fold {fold}: train={len(train)} clips/{len(train_users)} users; "
            f"outer-val={len(outer_val)} clips/{len(val_users)} users; "
            f"eligible classes={len(sampler.eligible_classes)}/{len(sampler.by_class_user)}; "
            f"pretrain batches={len(sampler)} x {sampler.batch_size}; "
            f"params={sum(p.numel() for p in model.parameters())}",
            flush=True,
        )
    print("DRY RUN PASSED: partitions are user-disjoint; no experiment artifacts were written.")


class _SyntheticDataset:
    def __init__(self):
        self.ids, self.labels, self.meta = [], {}, {}
        for label in range(4):
            for user in range(3):
                for example in range(2):
                    sid = f"c{label}_u{user}_e{example}"
                    self.ids.append(sid)
                    self.labels[sid] = label
                    self.meta[sid] = {"user": f"user{user}"}

    def __len__(self):
        return len(self.ids)


def smoke_test():
    seed_everything(123)
    dataset = _SyntheticDataset()
    sampler_a = CrossUserClassBatchSampler(
        dataset, classes_per_batch=4, samples_per_class=3, seed=19
    )
    sampler_b = CrossUserClassBatchSampler(
        dataset, classes_per_batch=4, samples_per_class=3, seed=19
    )
    batches_a, batches_b = list(sampler_a), list(sampler_b)
    if batches_a != batches_b:
        raise AssertionError("sampler is not deterministic")
    for batch in batches_a:
        if len(batch) != len(set(batch)):
            raise AssertionError("sampler repeated a clip despite sufficient unique clips")
        class_users = defaultdict(set)
        for index in batch:
            sid = dataset.ids[index]
            class_users[dataset.labels[sid]].add(dataset.meta[sid]["user"])
        if any(len(users) < 2 for users in class_users.values()):
            raise AssertionError("a sampled class lacks two distinct users")

    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    users = torch.tensor([0, 1, 2, 0, 1, 2])
    features = torch.randn(6, 12, requires_grad=True)
    loss = cross_user_supcon_loss(features, labels, users, temperature=0.1)
    loss.backward()
    if not torch.isfinite(loss) or features.grad is None:
        raise AssertionError("contrastive loss did not produce finite gradients")

    model = har_models.build("skel", in_ch=153, arch="multitcn", width=16)
    projector = ProjectionHead(model.feature_dim, hidden_dim=32, out_dim=12)
    parameters = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    encoded = model.forward_features(torch.randn(6, 32, 153))
    if encoded.shape != (6, model.feature_dim):
        raise AssertionError("unexpected MultiTCN feature shape")
    train_loss = cross_user_supcon_loss(
        projector(encoded), labels, users, temperature=0.1
    ) + nn.CrossEntropyLoss()(model.head(model.drop(encoded)), labels)
    optimizer_step(train_loss, optimizer, scaler, scheduler, parameters, grad_clip=5.0)
    if not torch.isfinite(train_loss):
        raise AssertionError("combined model/projector optimization step was not finite")

    state = {key: value.clone() for key, value in model.state_dict().items()}
    reloaded = har_models.build("skel", in_ch=153, arch="multitcn", width=16)
    reloaded.load_state_dict(state, strict=True)
    if any(key.startswith("net.") for key in state):
        raise AssertionError("projection-head parameters leaked into backbone checkpoint")
    print(
        f"SMOKE TEST PASSED: deterministic cross-user batches, finite loss "
        f"{float(loss.detach()):.4f}, feature_dim={model.feature_dim}, plain checkpoint reload."
    )


def parse_folds(text):
    try:
        folds = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--folds must be a comma-separated subset of 0,1,2,3") from exc
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(4) for fold in folds):
        raise ValueError("--folds must be a unique, non-empty subset of 0,1,2,3")
    return folds


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe cross-user supervised-contrastive pretraining followed "
            "by fixed-epoch natural-prior fine-tuning."
        )
    )
    parser.add_argument("--tag", default="skel_multitcn_xusupcon")
    parser.add_argument("--exp", default="EXP-040")
    parser.add_argument("--folds", default="0,2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument(
        "--pretrain-batches",
        type=int,
        default=0,
        help="batches per pretrain epoch (0 = ceil(train clips / batch size))",
    )
    parser.add_argument("--classes-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--projection-hidden", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--supcon-weight", type=float, default=1.0)
    parser.add_argument("--aux-ce-weight", type=float, default=0.5)
    parser.add_argument("--aug-spec", default="trunc,tjit")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--finetune-bs", type=int, default=64)
    parser.add_argument("--eval-bs", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="use CUDA float16 autocast")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate real folds/sampler/model without loading clips or writing files",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run synthetic sampler/loss/checkpoint tests without dataset access",
    )
    return parser.parse_args()


def validate_args(args):
    positive_ints = {
        "--width": args.width,
        "--pretrain-epochs": args.pretrain_epochs,
        "--finetune-epochs": args.finetune_epochs,
        "--classes-per-batch": args.classes_per_batch,
        "--samples-per-class": args.samples_per_class,
        "--projection-hidden": args.projection_hidden,
        "--projection-dim": args.projection_dim,
        "--finetune-bs": args.finetune_bs,
        "--eval-bs": args.eval_bs,
        "--log-every": args.log_every,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least 2")
    if args.pretrain_batches < 0 or args.workers < 0:
        raise ValueError("--pretrain-batches and --workers cannot be negative")
    if args.temperature <= 0 or args.lr <= 0 or args.finetune_lr <= 0:
        raise ValueError("temperature and learning rates must be positive")
    if args.supcon_weight <= 0 or args.aux_ce_weight < 0:
        raise ValueError("--supcon-weight must be positive and --aux-ce-weight non-negative")
    if not args.tag or not args.exp:
        raise ValueError("--tag and --exp cannot be empty")


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    validate_args(args)
    folds = parse_folds(args.folds)
    if args.dry_run:
        dry_run(args, folds)
        return

    started = time.time()
    fold_results = []
    for fold in folds:
        result = run_fold(args, fold)
        fold_results.append(result)
    elapsed_minutes = (time.time() - started) / 60.0
    row, oof_path, manifest_path = save_run_artifacts(
        args, folds, fold_results, elapsed_minutes
    )
    print(
        f"RESULT {args.exp} {args.tag}: mean {row['mean']:.4f} ± {row['std']:.4f}; "
        f"OOF {os.path.relpath(oof_path, ROOT)}; "
        f"manifest {os.path.relpath(manifest_path, ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
