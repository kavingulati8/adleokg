"""Assignment 1 implementations, reused unchanged by Assignments 2 and 3."""

from __future__ import annotations

from torch.utils.data import Dataset


def paired_flip(image, target, horizontal=False, vertical=False):
    """Apply the same requested flips to a [C,H,W] image and [H,W] target."""
    # TODO: preserve dtypes and target values, including the ignore value -1.
    import numpy as np
    import torch

    dims = []
    if horizontal:
        dims.append(-1)  # mirror left-right (width axis)
    if vertical:
        dims.append(-2)  # mirror up-down (height axis)

    def flip(array):
        if isinstance(array, torch.Tensor):
            return array.flip(dims) if dims else array.clone()
        array = np.asarray(array)
        return np.flip(array, axis=tuple(dims)).copy() if dims else array.copy()

    # Same flips on both, so every image pixel stays aligned with its label pixel.
    # Flipping only moves values, so dtypes, classes 0/1/2 and ignore value -1 are kept.
    return flip(image), flip(target)


def normalize_image(image, valid=None):
    """Return per-band float32 z-scores computed over valid image pixels only.

    Zero-fill invalid image pixels. Protect constant and all-invalid bands from
    NaN/Inf. Label availability must never influence image normalization.
    """
    # TODO: compute protected per-band statistics without modifying the input.
    import numpy as np
    import torch

    is_torch = isinstance(image, torch.Tensor)
    x = image.detach().cpu().numpy() if is_torch else np.asarray(image)
    x = x.astype(np.float32, copy=True)  # work on a copy: never modify the input

    finite = np.isfinite(x)
    if valid is None:
        valid_mask = finite
    else:
        v = valid.detach().cpu().numpy() if isinstance(valid, torch.Tensor) else np.asarray(valid)
        v = v.astype(bool)
        if v.ndim == x.ndim - 1:  # one [H,W] mask shared by all bands
            v = np.broadcast_to(v[None], x.shape)
        valid_mask = v & finite

    out = np.zeros_like(x, dtype=np.float32)  # invalid pixels stay 0
    for band in range(x.shape[0]):
        m = valid_mask[band]
        if not m.any():
            continue  # all-invalid band: leave zeros, no NaN
        values = x[band][m].astype(np.float64)
        mean = values.mean()
        std = values.std()
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0  # constant band: avoid divide-by-zero
        out[band][m] = ((values - mean) / std).astype(np.float32)

    return torch.from_numpy(out) if is_torch else out


def training_transform(image, target):
    """Apply paired stochastic augmentation after preprocessing."""
    # TODO: draw shared augmentation decisions and call paired_flip.
    import torch

    # One random decision per axis, shared by image AND target.
    horizontal = bool(torch.rand(1).item() < 0.5)
    vertical = bool(torch.rand(1).item() < 0.5)
    return paired_flip(image, target, horizontal=horizontal, vertical=vertical)


def evaluation_transform(image, target):
    """Return deterministic copies for validation and test data."""
    # TODO: validation/test must not use stochastic augmentation.
    import numpy as np
    import torch

    # Deterministic: return untouched copies, never random augmentation.
    def copy(array):
        if isinstance(array, torch.Tensor):
            return array.clone()
        return np.asarray(array).copy()

    return copy(image), copy(target)


def build_mnist_loaders(
    root,
    batch_size=64,
    seed=42,
    split_ids=None,
    limits=None,
    download=True,
):
    """Build reusable MNIST train/validate/test loaders.

    Split the packaged 60,000-image training pool into seeded 90/10 training
    and validation membership. Keep the official 10,000-image test pool held
    out. Use unique global IDs 0--59,999 for the training pool and
    60,000--69,999 for test. Apply stochastic augmentation only to training;
    all three splits use the documented MNIST mean 0.1307 and std 0.3081.
    Reconstruct exact membership when ``split_ids`` is supplied and return it
    with the three loaders. Optional limits are declared smoke-test budgets.
    """
    from torchvision import datasets, transforms

    # Training: small random rotation (like messy handwriting), then tensor + normalize.
    train_tf = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # Validation/test: same tensor + normalize, but NO randomness.
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Validation comes from the training pool but must not be rotated,
    # so that pool is loaded twice with different transforms.
    train_pool = datasets.MNIST(root, train=True, download=download, transform=train_tf)
    val_pool = datasets.MNIST(root, train=True, download=download, transform=eval_tf)
    test_pool = datasets.MNIST(root, train=False, download=download, transform=eval_tf)

    import torch
    from torch.utils.data import DataLoader, Subset

    # Step 3: membership as global IDs (train pool 0-59,999; test 60,000-69,999)
    if split_ids is None:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(60000, generator=generator).tolist()
        n_train = 54000  # 90% of 60,000
        split_ids = {
            "train": order[:n_train],
            "validate": order[n_train:],
            "test": list(range(60000, 70000)),
        }
    else:
        split_ids = {k: [int(i) for i in split_ids[k]] for k in ("train", "validate", "test")}

    # Limits are a runtime budget: they shrink what the loaders use, not the saved split
    limits = limits or {}

    def used(key):
        ids = split_ids[key]
        n = limits.get(key)
        return ids if n is None else ids[:n]

    # Step 4: loaders - only training is shuffled (with a seeded generator)
    train_loader = DataLoader(
        Subset(train_pool, used("train")),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_pool, used("validate")),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(test_pool, [i - 60000 for i in used("test")]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Step 5: return loaders and exact membership
    return {
        "train": train_loader,
        "validate": val_loader,
        "test": test_loader,
        "split_ids": split_ids,
    }


def build_eurosat_loaders(
    root,
    batch_size=16,
    seed=42,
    split_ids=None,
    limits=None,
    download=True,
):
    """Build reproducible EuroSAT100 train/validate/test loaders.

    Preserve EuroSAT100's documented 60/20/20 train/validation/test split.
    Return loaders under ``train``, ``validate`` and ``test`` plus ``split_ids``
    containing unique global integer membership from 0 through 99. If split_ids
    is supplied, reconstruct that membership exactly; A2 and A3 use this path
    instead of choosing new examples.
    Normalize images without fitting on validation or test data.
    Shuffle only training and use num_workers=0 for portable environments.
    Optional per-split limits are a predeclared debugging/compute budget.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchgeo.datasets import EuroSAT100

    # Step 1: the three official splits (torchgeo calls validation "val")
    root = str(root)
    official = {
        "train": EuroSAT100(root, split="train", download=download),
        "validate": EuroSAT100(root, split="val", download=download),
        "test": EuroSAT100(root, split="test", download=download),
    }

    # Step 2: global IDs 0-99 in official order (train 0-59, validate 60-79, test 80-99)
    lookup = []  # lookup[global_id] = (dataset, local index)
    default_ids = {}
    for key in ("train", "validate", "test"):
        start = len(lookup)
        lookup.extend((official[key], i) for i in range(len(official[key])))
        default_ids[key] = list(range(start, len(lookup)))

    if split_ids is None:
        split_ids = default_ids
    else:
        # Replay exact membership supplied by A2/A3
        split_ids = {k: [int(i) for i in split_ids[k]] for k in ("train", "validate", "test")}

    def load(global_id):
        dataset, local = lookup[global_id]
        return dataset[local]

    # Limits are a runtime budget for the loaders, not a new split
    limits = limits or {}

    def used(key):
        ids = split_ids[key]
        n = limits.get(key)
        return ids if n is None else ids[:n]

    # Step 3: per-band statistics fitted on TRAINING images only (no val/test leakage)
    train_stack = torch.stack([load(g)["image"].float() for g in split_ids["train"]])
    mean = train_stack.mean(dim=(0, 2, 3))[:, None, None]
    std = train_stack.std(dim=(0, 2, 3)).clamp_min(1e-6)[:, None, None]

    # Step 4: wrapper returning (normalized float32 image [13,H,W], int64 label)
    class _EuroSATSplit(Dataset):
        def __init__(self, ids):
            self.ids = list(ids)

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            sample = load(self.ids[index])
            image = ((sample["image"].float() - mean) / std).float()
            label = torch.as_tensor(sample["label"]).long()
            return image, label

    # Step 5: loaders - only training shuffled; num_workers=0 for portability
    train_loader = DataLoader(
        _EuroSATSplit(used("train")),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        _EuroSATSplit(used("validate")),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        _EuroSATSplit(used("test")),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return {
        "train": train_loader,
        "validate": val_loader,
        "test": test_loader,
        "split_ids": split_ids,
    }


def _read_planet_pair(image_path, mask_path):
    """Read one image/mask pair after checking CRS, transform and grid agree."""
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as img, rasterio.open(mask_path) as lbl:
        if img.crs != lbl.crs:
            raise ValueError(f"CRS mismatch: {image_path} vs {mask_path}")
        if not img.transform.almost_equals(lbl.transform):
            raise ValueError(f"Affine transform mismatch: {image_path} vs {mask_path}")
        if (img.width, img.height) != (lbl.width, lbl.height):
            raise ValueError(f"Grid size mismatch: {image_path} vs {mask_path}")
        if img.count < 4:
            raise ValueError(f"Expected 4 image bands, found {img.count}: {image_path}")
        image = img.read(indexes=[1, 2, 3, 4]).astype(np.float32)
        image_valid = img.dataset_mask() > 0
        label = lbl.read(1)
        label_valid = lbl.dataset_mask() > 0
    return image, image_valid, label, label_valid


class PlanetCatalogDataset(Dataset):
    """Lazy three-class Planet image/mask dataset based on catalog pairs.

    Filter exactly one split, read 4-band images and 1-band labels lazily,
    verify CRS/transform/grid, keep classes 0/1/2, invalid labels -> -1.
    Returns (float32 image [4,H,W], int64 target [H,W], absolute image path).
    """

    def __init__(self, catalog_path, data_root, split, transform=None,
                 selected_rows=None, max_items=None):
        import numpy as np
        import pandas as pd
        from pathlib import Path

        if split not in ("train", "validate", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        catalog = pd.read_csv(catalog_path)  # catalog only - no rasters opened
        in_split = np.flatnonzero(catalog["split"].to_numpy() == split)
        if selected_rows is None:
            rows = [int(r) for r in in_split]
        else:
            if isinstance(selected_rows, dict):
                selected_rows = selected_rows[split]
            allowed = set(int(r) for r in in_split)
            rows = [int(r) for r in selected_rows]
            outside = [r for r in rows if r not in allowed]
            if outside:
                raise ValueError(f"Rows {outside[:5]} are not in split {split!r}")
        if max_items is not None:
            rows = rows[:max_items]
        root = Path(data_root)
        self.split = split
        self.rows = rows
        self.transform = transform
        self.image_paths = [root / catalog.iloc[r]["window_b"] for r in rows]
        self.mask_paths = [root / catalog.iloc[r]["mask"] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import torch

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image, image_valid, label, label_valid = _read_planet_pair(image_path, mask_path)
        # Normalization uses IMAGE validity only - labels never influence it.
        pixel_valid = image_valid & np.isfinite(image).all(axis=0)
        normalized = normalize_image(image, pixel_valid)
        # Keep classes 0/1/2; anything invalid or unexpected becomes -1.
        target = label.astype(np.int64)
        target[~label_valid | ~np.isin(label, (0, 1, 2))] = -1
        image_t = torch.from_numpy(np.ascontiguousarray(normalized)).float()
        target_t = torch.from_numpy(np.ascontiguousarray(target)).long()
        if self.transform is not None:
            image_t, target_t = self.transform(image_t, target_t)
        return image_t.float(), target_t.long(), str(image_path.resolve())


def match_directory_pairs(image_dir, label_dir, allowed_image_names):
    """Independently match permitted image/mask filenames by site and date.

    ``AO0632385_2022-11.tif`` matches ``AO0632385_13723_2022-11.tif``.
    Reject missing or ambiguous matches. Do not use catalog rows to build the
    pairs, other than the explicit permitted image-name pool.
    """
    raise NotImplementedError("A1: implement independent directory pairing")


class PlanetDirectoryDataset(Dataset):
    """Lazy directory-pair dataset with the catalog-loading item contract."""

    def __init__(self, pairs, transform=None):
        raise NotImplementedError("A1: store validated directory pairs")

    def __len__(self):
        raise NotImplementedError(
            "A1: implement PlanetDirectoryDataset.__len__"
        )

    def __getitem__(self, index):
        raise NotImplementedError(
            "A1: reuse your raster reading and preprocessing"
        )


def build_planet_loaders(
    catalog_path,
    data_root,
    batch_size=2,
    seed=42,
    selected_rows=None,
    limits=None,
):
    """Build official-split Planet loaders and return exact row membership.

    Use stochastic transforms only for training. If ``selected_rows`` is given,
    reconstruct precisely those catalog row indices. Otherwise, select rows
    inside each supplied train/validate/test split, applying optional limits.
    Return loaders and ``selected_rows`` for the A1 data protocol.
    """
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    catalog = pd.read_csv(catalog_path)
    splits = ("train", "validate", "test")
    if selected_rows is None:
        rng = np.random.default_rng(seed)
        limits = limits or {}
        selected_rows = {}
        for split in splits:
            rows = np.flatnonzero(catalog["split"].to_numpy() == split)
            n = limits.get(split)
            if n is not None and n < len(rows):
                rows = np.sort(rng.choice(rows, size=n, replace=False))
            selected_rows[split] = [int(r) for r in rows]
    else:
        # Replay exact rows from the A1 protocol
        selected_rows = {s: [int(r) for r in selected_rows[s]] for s in splits}

    def make_loader(split):
        transform = training_transform if split == "train" else evaluation_transform
        dataset = PlanetCatalogDataset(catalog_path, data_root, split,
                                       transform=transform,
                                       selected_rows=selected_rows[split])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(seed) if split == "train" else None,
            num_workers=0,
        )

    return {
        "train": make_loader("train"),
        "validate": make_loader("validate"),
        "test": make_loader("test"),
        "selected_rows": selected_rows,
    }


def save_data_protocol(
    path,
    *,
    seed,
    eurosat_split_ids,
    planet_selected_rows,
    settings,
    mnist_split_ids=None,
):
    """Save portable membership for exact A2/A3 reconstruction."""
    raise NotImplementedError("A1: implement save_data_protocol")


def load_data_protocol(path):
    """Load and validate the A1 protocol without silently inventing defaults."""
    raise NotImplementedError("A1: implement load_data_protocol")


# --- A1 Task 4: match_directory_pairs (student implementation) ---
def match_directory_pairs(image_dir, label_dir, allowed_image_names):
    """Pair each allowed image with exactly one mask by (site, acquisition date).

    Image and mask stems differ (mask names carry an extra ID), so pairs are
    matched on site + date, never on equal stems. Missing or ambiguous masks
    are rejected.
    """
    import re
    from pathlib import Path

    def site_and_date(path):
        stem = Path(path).stem
        site = stem.split("_")[0]
        match = re.search(r"\d{4}-\d{2}(?:-\d{2})?", stem)
        return site, (match.group(0) if match else None)

    image_dir, label_dir = Path(image_dir), Path(label_dir)
    allowed = {Path(name).name for name in allowed_image_names}

    labels_by_key = {}
    for label in sorted(label_dir.glob("*.tif")):
        labels_by_key.setdefault(site_and_date(label), []).append(label)

    pairs = []
    for name in sorted(allowed):
        image = image_dir / name
        if not image.exists():
            continue                      # image not on disk -> reject
        candidates = labels_by_key.get(site_and_date(image), [])
        if len(candidates) != 1:
            continue                      # missing or ambiguous mask -> reject
        pairs.append((image, candidates[0]))
    return pairs


# --- A1 Task 4: PlanetDirectoryDataset (student implementation) ---
class PlanetDirectoryDataset(Dataset):
    """Lazy directory-pair dataset with the catalog-loading item contract."""

    def __init__(self, pairs, transform=None):
        from pathlib import Path
        self.image_paths, self.mask_paths = [], []
        for image_path, mask_path in pairs:
            image_path, mask_path = Path(image_path), Path(mask_path)
            if not image_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"missing pair: {image_path}, {mask_path}")
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Reuse the exact catalog reading + preprocessing (CRS/transform checks,
        # image-only normalization, -1 for invalid labels, same return format).
        return PlanetCatalogDataset.__getitem__(self, index)


# --- A1 Task 5: save_data_protocol (student implementation) ---
def save_data_protocol(path, *, seed, eurosat_split_ids, planet_selected_rows,
                       settings, mnist_split_ids=None):
    """Save portable membership for exact A2/A3 reconstruction."""
    import json, os
    from pathlib import Path, PureWindowsPath

    def portable(value):
        if hasattr(value, "tolist"):              # numpy arrays/scalars, tensors
            value = value.tolist()
        if isinstance(value, dict):
            return {str(k): portable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(v) for v in value]
        if isinstance(value, set):
            return sorted(portable(v) for v in value)
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and (os.path.isabs(value)
                                       or PureWindowsPath(value).is_absolute()):
            raise ValueError(f"absolute path not allowed in protocol: {value}")
        return value

    protocol = {
        "seed": int(seed),
        "mnist_split_ids": portable(mnist_split_ids),
        "eurosat_split_ids": portable(eurosat_split_ids),
        "planet_selected_rows": portable(planet_selected_rows),
        "settings": portable(settings),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True))
    return path