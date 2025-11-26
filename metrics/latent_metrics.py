import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class DecoderTrainConfig:
    """
    Config for training latent -> action decoders.
    """
    batch_size: int = 512
    num_epochs: int = 10          # "a couple of epochs" – feel free to set this lower/higher
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_split: float = 0.1        # fraction of data used for validation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    shuffle: bool = True


class LatentToActionDecoder(torch.nn.Module):
    """
    Simple MLP that decodes a latent vector (z_a or z_q)
    back to the continuous action space.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_single_decoder(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: DecoderTrainConfig,
    model: Optional[LatentToActionDecoder] = None,
) -> Tuple[LatentToActionDecoder, Dict[str, List[float]]]:
    """
    Train a single latent->action decoder with MSE loss.

    x: (N, D_latent)
    y: (N, D_action)
    model:
        If provided, continue training this model.
        If None, a fresh LatentToActionDecoder will be created.
    """
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(cfg.device)
    x = x.to(device)
    y = y.to(device)

    N = x.shape[0]
    if N == 0:
        raise ValueError("No samples available to train decoder.")

    # --- train/val split ---
    n_val = int(N * cfg.val_split)
    if n_val > 0:
        idx = torch.randperm(N, device=device)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
    else:
        x_train, y_train = x, y
        x_val, y_val = x[:0], y[:0]

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        drop_last=False,
    )

    # --- (re-)initialize model if needed ---
    if model is None:
        model = LatentToActionDecoder(
            input_dim=x.shape[1],
            action_dim=y.shape[1],
        )
    model = model.to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for xb, yb in train_loader:
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_batches += 1

        mean_train_loss = total_loss / max(1, total_batches)
        history["train_loss"].append(mean_train_loss)

        # --- validation ---
        if n_val > 0 and x_val.numel() > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = loss_fn(val_pred, y_val).item()
        else:
            val_loss = float("nan")
        history["val_loss"].append(val_loss)

    return model, history


def train_latent_decoders_from_aggregator(
    agg: "LatentMetricsAggregator",
    cfg: Optional[DecoderTrainConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains two MLP decoders that map z_a and z_q to actions
    using the data accumulated in a LatentMetricsAggregator.

    Parameters
    ----------
    agg : LatentMetricsAggregator
        Must have been fed with data via agg.update(...).
    cfg : DecoderTrainConfig, optional
        Training hyperparameters.

    Returns
    -------
    models : dict
        {
            "z_a": LatentToActionDecoder,
            "z_q": LatentToActionDecoder,
        }
    histories : dict
        {
            "z_a": {"train_loss": [...], "val_loss": [...]},
            "z_q": {"train_loss": [...], "val_loss": [...]},
        }
    """
    if cfg is None:
        cfg = DecoderTrainConfig()

    # Extract numpy arrays from the aggregator
    z_a_np = agg._stack(agg.latent_actions)
    z_q_np = agg._stack(agg.quantized_actions)
    act_np = agg._stack(agg.actions)

    if z_a_np.size == 0 or z_q_np.size == 0 or act_np.size == 0:
        raise ValueError(
            "LatentMetricsAggregator does not contain any samples. "
            "Make sure to call agg.update(...) before training decoders."
        )

    # Convert to tensors
    z_a = torch.from_numpy(z_a_np).float()
    z_q = torch.from_numpy(z_q_np).float()
    actions = torch.from_numpy(act_np).float()

    models: Dict[str, Any] = {}
    histories: Dict[str, Any] = {}

    # z_a -> action
    model_a, hist_a = _train_single_decoder(z_a, actions, cfg)
    models["z_a"] = model_a
    histories["z_a"] = hist_a

    # z_q -> action
    model_q, hist_q = _train_single_decoder(z_q, actions, cfg)
    models["z_q"] = model_q
    histories["z_q"] = hist_q

    # z_q split -> action split (per-code decoder)
    num_splits = agg.config.num_splits
    if z_q.shape[1] % num_splits != 0:
        raise ValueError("z_q dimension must be divisible by num_splits for split decoder")
    if z_a.shape[1] % num_splits != 0:
        raise ValueError("z_a dimension must be divisible by num_splits for split decoder")
    if actions.shape[1] % num_splits != 0:
        raise ValueError("Action dimension must be divisible by num_splits for split decoder")

    z_q_chunk_dim = z_q.shape[1] // num_splits
    z_a_chunk_dim = z_a.shape[1] // num_splits
    act_chunk_dim = actions.shape[1] // num_splits

    z_q_split = z_q.reshape(-1, num_splits, z_q_chunk_dim).reshape(-1, z_q_chunk_dim)
    z_a_split = z_a.reshape(-1, num_splits, z_a_chunk_dim).reshape(-1, z_a_chunk_dim)
    act_split = actions.reshape(-1, num_splits, act_chunk_dim).reshape(-1, act_chunk_dim)

    model_q_split, hist_q_split = _train_single_decoder(z_q_split, act_split, cfg)
    model_a_split, hist_a_split = _train_single_decoder(z_a_split, act_split, cfg)
    models["z_q_split"] = model_q_split
    models["z_a_split"] = model_a_split
    histories["z_q_split"] = hist_q_split
    histories["z_a_split"] = hist_a_split

    return models, histories

def _pairwise_chebyshev(x: np.ndarray) -> np.ndarray:
    diff = np.abs(x[:, None, :] - x[None, :, :])
    dist = diff.max(axis=2)
    np.fill_diagonal(dist, 0.0)
    return dist


def _pairwise_l2(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist


def _sample_with_indices(
    x: np.ndarray, max_samples: Optional[int], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or x.shape[0] <= max_samples:
        return x, np.arange(x.shape[0])
    indices = rng.permutation(x.shape[0])[:max_samples]
    return x[indices], indices


def _digamma(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64)
    return torch.digamma(tensor).item()


def _digamma_vector(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float64))
    return torch.digamma(tensor).cpu().numpy()


def knn_mutual_information(x: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator) -> float:
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return float("nan")
    if n <= k:
        return float("nan")
    x = x[:n]
    y = y[:n]
    if n > 10000:
        indices = rng.permutation(n)[:10000]
        x = x[indices]
        y = y[indices]

    xy = np.concatenate([x, y], axis=1)
    dist_xy = _pairwise_chebyshev(xy)
    np.fill_diagonal(dist_xy, np.inf)
    kth = np.partition(dist_xy, kth=k - 1, axis=1)[:, k - 1]
    kth = kth + 1e-10

    dist_x = _pairwise_chebyshev(x)
    dist_y = _pairwise_chebyshev(y)

    np.fill_diagonal(dist_x, np.inf)
    np.fill_diagonal(dist_y, np.inf)

    nx = (dist_x < kth[:, None]).sum(axis=1)
    ny = (dist_y < kth[:, None]).sum(axis=1)
    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    return _digamma(k) + _digamma(n) - float(np.mean(_digamma_vector(nx + 1) + _digamma_vector(ny + 1)))


def knn_consistency(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_samples: int
) -> float:

    if features.shape[0] <= k:
        return float("nan")

    if labels.ndim == 2:
        composite = []
        for row in labels:
            vals, counts = np.unique(row, return_counts=True)
            maxc = counts.max()
            majority = vals[counts == maxc]
            if majority.size == 1:
                composite.append(majority[0])
            else:
                composite.append(row[0])
        labels = np.array(composite, dtype=int)

    features, indices = _sample_with_indices(features, max_samples, rng)
    labels = labels[indices]

    if features.shape[0] <= k:
        return float("nan")

    dist = _pairwise_l2(features)
    np.fill_diagonal(dist, np.inf)

    nn_indices = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    neighbor_labels = labels[nn_indices]

    matches = (neighbor_labels == labels[:, None]).mean(axis=1)
    return float(np.mean(matches))




def compute_class_variances(features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if features.size == 0 or labels.size == 0:
        return float("nan"), float("nan")

    if labels.ndim == 2:
        composite = []
        for row in labels:
            vals, counts = np.unique(row, return_counts=True)
            maxc = counts.max()
            majority = vals[counts == maxc]
            if majority.size == 1:
                composite.append(majority[0])
            else:
                composite.append(row[0])
        labels = np.array(composite, dtype=int)

    unique_labels = np.unique(labels)
    class_means = []
    intra_vars = []

    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue
        feats = features[mask]
        intra_vars.append(feats.var(axis=0, ddof=1).mean())
        class_means.append(feats.mean(axis=0))

    if not class_means:
        return float("nan"), float("nan")

    class_means = np.stack(class_means)
    intra_val = float(np.mean(intra_vars)) if intra_vars else float("nan")

    inter = _pairwise_l2(class_means)
    triu = np.triu_indices_from(inter, k=1)
    inter_val = float(np.mean(inter[triu] ** 2)) if triu[0].size else float("nan")

    return intra_val, inter_val



def compute_fdr(intra: float, inter: float) -> float:
    if math.isnan(intra) or intra == 0:
        return float("nan")
    return inter / intra if not math.isnan(inter) else float("nan")


@dataclass
class LatentMetricsConfig:
    num_codes: int
    max_samples: int = 4096
    mi_k: int = 5
    knn_k: int = 10
    action_round_decimals: int = 3
    umap_points: int = 2000
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    seed: int = 0
    num_splits: int = 5


class LatentMetricsAggregator:
    def __init__(
        self,
        config: LatentMetricsConfig,
        decoder_cfg: Optional[DecoderTrainConfig] = None,
    ):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # --- decoder / training config ---
        self.decoder_cfg = decoder_cfg or DecoderTrainConfig()
        self.z_a_decoder: Optional[LatentToActionDecoder] = None
        self.z_q_decoder: Optional[LatentToActionDecoder] = None
        self.z_q_split_decoder: Optional[LatentToActionDecoder] = None
        self.z_a_split_decoder: Optional[LatentToActionDecoder] = None
        # full training histories per decoder
        self.decoder_histories: Dict[str, Dict[str, List[float]]] = {}

        self.action_lookup: Dict[Tuple[float, ...], int] = {}
        self.reset()
        self.centroids = np.array([
            [ 10.012674  ,  10.605386  ],
            [-47.39698   ,  -9.770397  ],
            [-20.167576  ,   0.45612565],
            [ 20.962193  , -17.880518  ],
            [  6.682827  ,  42.526566  ],
            [ -6.6809154 , -37.056248  ],
            [ 50.51583   ,  10.196131  ],
            [-15.197311  ,  23.22058   ],
            [ -1.4817983 ,  -6.2199874 ],
        ], dtype=np.float32)

        # Apply same scaling as dataset
        self.centroids /= 100.0

    def _train_action_decoders(
        self,
        z_a: np.ndarray,
        z_q: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, float]:
            """
            Train latent->action MLP decoders on all accumulated samples.

            Stores:
                self.z_a_decoder, self.z_q_decoder
                self.decoder_histories = { "z_a": history, "z_q": history }

            Returns a dict of scalar metrics (e.g. last-epoch losses) that
            can be merged into the overall metrics dict in `compute()`.
            """
            metrics: Dict[str, float] = {}

            if z_a.size == 0 or z_q.size == 0 or actions.size == 0:
                return metrics

            # Convert to tensors
            z_a_t = torch.from_numpy(z_a).float()
            z_q_t = torch.from_numpy(z_q).float()
            act_t = torch.from_numpy(actions).float()

            # Lazily create decoders if we don't know dims until now
            if self.z_a_decoder is None:
                self.z_a_decoder = LatentToActionDecoder(
                    input_dim=z_a_t.shape[1],
                    action_dim=act_t.shape[1],
                )
            if self.z_q_decoder is None:
                self.z_q_decoder = LatentToActionDecoder(
                    input_dim=z_q_t.shape[1],
                    action_dim=act_t.shape[1],
                )
            if self.z_q_split_decoder is None:
                assert (
                    z_q_t.shape[1] % self.config.num_splits == 0
                ), "z_q dimensionality must be divisible by num_splits"
                assert (
                    act_t.shape[1] % self.config.num_splits == 0
                ), "Action dimensionality must be divisible by num_splits"

                self.z_q_split_decoder = LatentToActionDecoder(
                    input_dim=z_q_t.shape[1] // self.config.num_splits,
                    action_dim=act_t.shape[1] // self.config.num_splits,
                )
            if self.z_a_split_decoder is None:
                assert (
                    z_a_t.shape[1] % self.config.num_splits == 0
                ), "z_a dimensionality must be divisible by num_splits"
                assert (
                    act_t.shape[1] % self.config.num_splits == 0
                ), "Action dimensionality must be divisible by num_splits"

                self.z_a_split_decoder = LatentToActionDecoder(
                    input_dim=z_a_t.shape[1] // self.config.num_splits,
                    action_dim=act_t.shape[1] // self.config.num_splits,
                )

            # Train / continue training
            self.z_a_decoder, hist_a = _train_single_decoder(
                z_a_t, act_t, self.decoder_cfg, model=self.z_a_decoder
            )
            self.z_q_decoder, hist_q = _train_single_decoder(
                z_q_t, act_t, self.decoder_cfg, model=self.z_q_decoder
            )

            z_q_split = z_q_t.reshape(
                -1, self.config.num_splits, z_q_t.shape[1] // self.config.num_splits
            ).reshape(-1, z_q_t.shape[1] // self.config.num_splits)
            z_a_split = z_a_t.reshape(
                -1, self.config.num_splits, z_a_t.shape[1] // self.config.num_splits
            ).reshape(-1, z_a_t.shape[1] // self.config.num_splits)
            act_split = act_t.reshape(
                -1, self.config.num_splits, act_t.shape[1] // self.config.num_splits
            ).reshape(-1, act_t.shape[1] // self.config.num_splits)

            self.z_q_split_decoder, hist_q_split = _train_single_decoder(
                z_q_split, act_split, self.decoder_cfg, model=self.z_q_split_decoder
            )
            self.z_a_split_decoder, hist_a_split = _train_single_decoder(
                z_a_split, act_split, self.decoder_cfg, model=self.z_a_split_decoder
            )

            self.decoder_histories = {
                "z_a": hist_a,
                "z_q": hist_q,
                "z_q_split": hist_q_split,
                "z_a_split": hist_a_split,
            }

            # Use last-epoch MSE as basic scalar metrics for now
            metrics["z_a_decoder_train_mse"] = hist_a["train_loss"][-1]
            metrics["z_a_decoder_val_mse"] = hist_a["val_loss"][-1]
            metrics["z_q_decoder_train_mse"] = hist_q["train_loss"][-1]
            metrics["z_q_decoder_val_mse"] = hist_q["val_loss"][-1]
            metrics["z_q_split_decoder_train_mse"] = hist_q_split["train_loss"][-1]
            metrics["z_q_split_decoder_val_mse"] = hist_q_split["val_loss"][-1]
            metrics["z_a_split_decoder_train_mse"] = hist_a_split["train_loss"][-1]
            metrics["z_a_split_decoder_val_mse"] = hist_a_split["val_loss"][-1]

            return metrics


    def reset(self) -> None:
        self.latent_actions: List[np.ndarray] = []
        self.quantized_actions: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.state_curr: List[np.ndarray] = []
        self.state_next: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.code_indices: List[np.ndarray] = []
        self.code_labels: List[np.ndarray] = []
        self.pred_errors: List[np.ndarray] = []
        self.pred_error_labels: List[np.ndarray] = []

    def _actions_to_labels(self, actions: np.ndarray) -> np.ndarray:
        """
        Convert stacked 2-D centroid actions into their centroid indices.
        Works with arbitrary frameskip (inferred from shape).
        Input:
            actions: (N, frameskip*2)
        Output:
            labels: (N, frameskip) with integer centroid IDs in {0..8}
        """
        base_action_dim = 2
        frameskip = actions.shape[1] // base_action_dim
        actions = actions.reshape(-1, frameskip, base_action_dim)
        N = actions.shape[0]
        labels = np.zeros((N, frameskip), dtype=np.int64)
        for i in range(frameskip):
            sub = actions[:, i, :]                         # (N,2)
            diff = sub[:, None, :] - self.centroids[None, :, :]  # (N,9,2)
            dist = np.sum(diff * diff, axis=2)                    # (N,9)
            labels[:, i] = np.argmin(dist, axis=1)                # best centroid
        return labels

    def update(
        self,
        latent_actions: torch.Tensor,
        quantized_actions: torch.Tensor,
        actions: torch.Tensor,
        state_repr: torch.Tensor,
        code_indices: torch.Tensor,
        per_step_errors: Optional[torch.Tensor] = None,
        per_step_error_actions: Optional[torch.Tensor] = None,
    ) -> None:
        with torch.no_grad():
            assert latent_actions.ndim >= 3, "latent_actions must be [B, T, ...]"
            assert quantized_actions.ndim >= 3, "quantized_actions must be [B, T, ...]"
            assert actions.ndim >= 3, "actions must be [B, T, ...]"
            assert state_repr.ndim >= 3, "state_repr must be [B, T, ...]"
            assert code_indices.ndim >= 3, "code_indices must be [B, T, num_splits]"
            assert (
                latent_actions.shape[0]
                == quantized_actions.shape[0]
                == actions.shape[0]
                == state_repr.shape[0]
                == code_indices.shape[0]
            ), "All inputs must share the same batch dimension"

            if latent_actions.ndim == 4:
                latent_actions = latent_actions.squeeze(2)
            if quantized_actions.ndim == 4:
                quantized_actions = quantized_actions.squeeze(2)

            time_dim = min(
                latent_actions.shape[1],
                quantized_actions.shape[1],
                actions.shape[1],
                state_repr.shape[1],
                code_indices.shape[1],
            )

            # The final action token is a placeholder; restrict aggregation to the
            # first t-1 steps to keep representations, codes, and ground-truth
            # actions aligned.
            if time_dim < 2:
                return

            valid_steps = time_dim - 1
            assert code_indices.shape[1] >= valid_steps, "code_indices too short for valid steps"

            z_a = (
                latent_actions[:, :valid_steps, :]
                .reshape(-1, latent_actions.shape[-1])
                .cpu()
                .numpy()
            )
            z_q = (
                quantized_actions[:, :valid_steps, :]
                .reshape(-1, quantized_actions.shape[-1])
                .cpu()
                .numpy()
            )
            act_curr = (
                actions[:, :valid_steps, :]
                .reshape(-1, actions.shape[-1])
                .cpu()
                .numpy()
            )
            state_t = (
                state_repr[:, :valid_steps, :]
                .reshape(-1, state_repr.shape[-1])
                .cpu()
                .numpy()
            )
            state_tp1 = (
                state_repr[:, 1 : valid_steps + 1, :]
                .reshape(-1, state_repr.shape[-1])
                .cpu()
                .numpy()
            )
            codes = code_indices[:, :valid_steps]                 # (B, valid_steps, frameskip)
            labels = self._actions_to_labels(act_curr)            # (M, frameskip)
            labels = labels.reshape(-1)                           # (M * frameskip,)

            frameskip = codes.shape[-1]
            assert frameskip == self.config.num_splits, "Unexpected number of splits in codes"
            assert act_curr.shape[1] % frameskip == 0, "Action dimension not divisible by num_splits"
            assert latent_actions.shape[-1] % frameskip == 0, "Latent dim not divisible by num_splits"
            assert quantized_actions.shape[-1] % frameskip == 0, "Quantized latent dim not divisible by num_splits"
            assert codes.shape[0] * codes.shape[1] * frameskip == labels.shape[0], "Codes and labels must align"

            # Flatten codes directly; frameskip == num_splits
            codes_flat = codes.reshape(-1).cpu().numpy()          # (M * frameskip,)

            self.latent_actions.append(z_a)
            self.quantized_actions.append(z_q)
            self.actions.append(act_curr)
            self.state_curr.append(state_t)
            self.state_next.append(state_tp1)
            self.labels.append(labels)
            self.code_indices.append(codes_flat)
            self.code_labels.append(labels)

            if per_step_errors is not None and per_step_error_actions is not None:
                error_steps = min(valid_steps, per_step_errors.shape[1], per_step_error_actions.shape[1])
                errors = per_step_errors[:, :error_steps].reshape(-1).cpu().numpy()
                error_actions = (
                    per_step_error_actions[:, :error_steps, :]
                    .reshape(-1, per_step_error_actions.shape[-1])
                    .cpu()
                    .numpy()
                )
                error_labels = self._actions_to_labels(error_actions)
                self.pred_errors.append(errors)
                self.pred_error_labels.append(error_labels)

    def _stack(self, tensors: List[np.ndarray]) -> np.ndarray:
        if not tensors:
            return np.empty((0,))
        return np.concatenate(tensors, axis=0)

    def _compute_code_usage(self, codes: np.ndarray) -> Dict[str, float]:
        hist = np.bincount(codes, minlength=self.config.num_codes)
        total = hist.sum()
        probs = hist / max(total, 1)
        mask = probs > 0
        entropy_bits = -np.sum(probs[mask] * np.log2(probs[mask])) if mask.any() else 0.0
        perplexity = 2 ** entropy_bits
        dead_fraction = float(np.sum(hist == 0) / self.config.num_codes)
        return {
            "code_perplexity": float(perplexity),
            "dead_code_fraction": dead_fraction,
            "bits_per_action": float(entropy_bits),
        }


    def _compute_action_action_overlap(self, codes: np.ndarray, labels: np.ndarray):
        """
        Computes an action-action Jaccard similarity heatmap.
        Returns only the matplotlib Figure for logging (e.g., to W&B).
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        # Collect codes used per action
        per_action_codes = defaultdict(set)
        for code, label in zip(codes.tolist(), labels.tolist()):
            per_action_codes[int(label)].add(int(code))

        # Sort action identifiers for consistent ordering
        actions = sorted(per_action_codes.keys())
        A = len(actions)

        # Build the action–action Jaccard similarity matrix
        overlap_matrix = np.zeros((A, A), dtype=float)
        for i, ai in enumerate(actions):
            Ci = per_action_codes[ai]
            for j, aj in enumerate(actions):
                Cj = per_action_codes[aj]
                union = len(Ci | Cj)
                inter = len(Ci & Cj)
                overlap_matrix[i, j] = (inter / union) if union > 0 else 0.0

        # ---- Create heatmap figure ----
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(overlap_matrix, cmap="viridis", vmin=0, vmax=1)

        ax.set_xticks(np.arange(A))
        ax.set_yticks(np.arange(A))
        ax.set_xticklabels(actions)
        ax.set_yticklabels(actions)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Action")
        ax.set_ylabel("Action")
        ax.set_title("Action–Action Code Usage Overlap (Jaccard Similarity)")

        fig.colorbar(im, ax=ax, label="Similarity")
        fig.tight_layout()

        return fig

    def _compute_usage_overlap(self, codes: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if codes.size == 0 or labels.size == 0:
            return {}

        unique_labels = np.unique(labels.astype(int))
        per_action_codes = {
            int(lbl): set(codes[labels == lbl].astype(int).tolist()) for lbl in unique_labels
        }

        overlaps: List[float] = []
        action_code_counts: List[int] = []

        for i, li in enumerate(unique_labels):
            Ci = per_action_codes[int(li)]
            action_code_counts.append(len(Ci))
            for lj in unique_labels[i + 1 :]:
                Cj = per_action_codes[int(lj)]
                union = len(Ci | Cj)
                inter = len(Ci & Cj)
                if union > 0:
                    overlaps.append(inter / union)

        mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")
        mean_codes_per_action = (
            float(np.mean(action_code_counts)) if action_code_counts else float("nan")
        )

        return {
            "code_action_jaccard_mean": mean_overlap,
            "codes_per_action_mean": mean_codes_per_action,
        }


    def _compute_umap_triplet(
    self,
    z_a: np.ndarray,
    z_q: np.ndarray,
    labels: np.ndarray,
) -> Tuple[
    Optional["matplotlib.figure.Figure"],
    Optional["matplotlib.figure.Figure"],
    Optional["matplotlib.figure.Figure"],
]:

        assert labels.ndim == 1
        assert (
            labels.shape[0] % self.config.num_splits == 0
        ), "Labels length must align with num_splits"

        N_full = labels.shape[0] // self.config.num_splits
        labels_5 = labels.reshape(N_full, self.config.num_splits)

        labels_composite = []
        for row in labels_5:
            vals, counts = np.unique(row, return_counts=True)
            max_count = counts.max()
            majority = vals[counts == max_count]
            if majority.size == 1:
                labels_composite.append(majority[0])
            else:
                labels_composite.append(row[0])
        labels_composite = np.array(labels_composite, dtype=int)

        assert z_a.shape[0] == N_full, "z_a rows must match number of action stacks"
        assert z_q.shape[0] == N_full, "z_q rows must match number of action stacks"

        N = min(N_full, self.config.umap_points)
        if N_full <= N:
            idx = np.arange(N_full)
        else:
            idx = self.rng.permutation(N_full)[:N]

        z_a_s = z_a[idx]
        z_q_s = z_q[idx]
        lbl_comp_s = labels_composite[idx]

        scaler_a = StandardScaler().fit(z_a_s)
        z_a_s = scaler_a.transform(z_a_s)

        scaler_q = StandardScaler().fit(z_q_s)
        z_q_s = scaler_q.transform(z_q_s)

        chunk_dim = z_q.shape[1] // self.config.num_splits
        z_q_split = z_q_s.reshape(N, self.config.num_splits, chunk_dim).reshape(
            N * self.config.num_splits, chunk_dim
        )
        lbl_split = labels_5[idx].reshape(-1)

        try:
            import matplotlib.pyplot as plt
            import umap
        except Exception:
            return None, None, None

        reducer = umap.UMAP(
            n_neighbors=self.config.umap_neighbors,
            min_dist=self.config.umap_min_dist,
            random_state=self.config.seed,
            metric="euclidean",
        )

        emb_a = reducer.fit_transform(z_a_s)
        emb_q = reducer.fit_transform(z_q_s)
        emb_q_split = reducer.fit_transform(z_q_split)

        fig_a, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(emb_a[:, 0], emb_a[:, 1], c=lbl_comp_s, cmap="tab10", s=6, alpha=0.8)
        ax.set_title("UMAP z_a")
        fig_a.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig_a.tight_layout()

        fig_q, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(emb_q[:, 0], emb_q[:, 1], c=lbl_comp_s, cmap="tab10", s=6, alpha=0.8)
        ax.set_title("UMAP z_q")
        fig_q.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig_q.tight_layout()

        fig_qs, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(emb_q_split[:, 0], emb_q_split[:, 1], c=lbl_split, cmap="tab10", s=6, alpha=0.8)
        ax.set_title("UMAP z_q split")
        fig_qs.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig_qs.tight_layout()

        return fig_a, fig_q, fig_qs

    def _build_confusion_heatmap(
        self, codes: np.ndarray, labels: np.ndarray
    ) -> Optional["matplotlib.figure.Figure"]:

        if codes.size == 0 or labels.size == 0:
            return None

        num_actions = int(labels.max()) + 1
        confusion = np.zeros((self.config.num_codes, num_actions), dtype=float)

        for code, label in zip(codes.tolist(), labels.tolist()):
            if 0 <= code < self.config.num_codes and 0 <= label < num_actions:
                confusion[code, label] += 1

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))

        im = ax.imshow(confusion, cmap="magma", aspect="auto")

        ax.set_xticks(np.arange(num_actions))
        ax.set_yticks(np.arange(self.config.num_codes))

        ax.set_xticklabels([str(i) for i in range(num_actions)])
        ax.set_yticklabels([str(i) for i in range(self.config.num_codes)])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Primitive Action")
        ax.set_ylabel("Code Index")
        ax.set_title("Code–Action Confusion")

        fig.colorbar(im, ax=ax, label="Count")
        fig.tight_layout()

        return fig


    def compute(self) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, float]]], Dict[str, object]]:
        metrics: Dict[str, float] = {}
        tables: Dict[str, List[Dict[str, float]]] = {}
        figures: Dict[str, object] = {}

        if not self.latent_actions:
            return metrics, tables, figures

        z_a = self._stack(self.latent_actions)
        z_q = self._stack(self.quantized_actions)
        actions = self._stack(self.actions)
        state_t = self._stack(self.state_curr)
        state_tp1 = self._stack(self.state_next)
        labels = self._stack(self.labels)
        codes = self._stack(self.code_indices).astype(int)
        code_labels = self._stack(self.code_labels) if self.code_labels else np.empty((0,))

        assert z_a.shape[0] == z_q.shape[0] == actions.shape[0]
        assert state_t.shape[0] == state_tp1.shape[0] == z_a.shape[0]
        assert (
            labels.shape[0] == z_a.shape[0] * self.config.num_splits
        ), "Label count must equal number of samples times num_splits"
        assert codes.shape[0] == labels.shape[0], "Codes and labels must have matching length"

        metrics.update(self._compute_code_usage(codes))

        intra_a, inter_a = compute_class_variances(z_a, labels)
        intra_q, inter_q = compute_class_variances(z_q, labels)
        metrics.update(
            {
                "z_a_intra_class_var": intra_a,
                "z_a_inter_class_var": inter_a,
                "z_q_intra_class_var": intra_q,
                "z_q_inter_class_var": inter_q,
                "z_a_fdr": compute_fdr(intra_a, inter_a),
                "z_q_fdr": compute_fdr(intra_q, inter_q),
            }
        )

        metrics.update(
            {
                "z_a_knn_consistency": knn_consistency(
                    z_a, labels, self.config.knn_k, self.rng, self.config.max_samples
                ),
                "z_q_knn_consistency": knn_consistency(
                    z_q, labels, self.config.knn_k, self.rng, self.config.max_samples
                ),
            }
        )

        metrics.update(
            {
                "mi_z_a_action": knn_mutual_information(z_a, actions, self.config.mi_k, self.rng),
                "mi_z_a_state_t": knn_mutual_information(z_a, state_t, self.config.mi_k, self.rng),
                "mi_z_a_state_tp1": knn_mutual_information(z_a, state_tp1, self.config.mi_k, self.rng),
                "mi_z_q_action": knn_mutual_information(z_q, actions, self.config.mi_k, self.rng),
                "mi_z_q_state_t": knn_mutual_information(z_q, state_t, self.config.mi_k, self.rng),
                "mi_z_q_state_tp1": knn_mutual_information(z_q, state_tp1, self.config.mi_k, self.rng),
            }
        )

        overall_error, per_action_error = self._compute_per_action_errors()
        metrics["next_state_error"] = overall_error
        error_fig = self._plot_bar_chart(
            per_action_error, "Per-Action Next-State Error", "Mean Error"
        )
        if error_fig is not None:
            figures["next_state_error_per_action"] = error_fig

        confusion_fig, label_stats = self._build_confusion_heatmap(codes, code_labels)
        if confusion_fig is not None:
            figures["code_action_confusion"] = confusion_fig
        metrics.update(label_stats)
        metrics.update(self._compute_usage_overlap(codes, code_labels))

        fig_a, fig_q, fig_qs = self._compute_umap_triplet(z_a, z_q, labels)
        if fig_a is not None:
            figures["umap_z_a"] = fig_a
        if fig_q is not None:
            figures["umap_z_q"] = fig_q
        if fig_qs is not None:
            figures["umap_z_q_split"] = fig_qs

        # --- Train latent->action decoders on accumulated data ---
        decoder_metrics = self._train_action_decoders(z_a, z_q, actions)
        metrics.update(decoder_metrics)
        
        return metrics, tables, figures
