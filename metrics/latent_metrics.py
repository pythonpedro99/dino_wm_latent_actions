import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class LatentToActionDecoder(torch.nn.Module):
    """
    Simple MLP that decodes a latent vector (z_a or z_q) back to the
    continuous action space. The module stores its own training hyperparameters
    and exposes a ``fit`` method for reuse.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        batch_size: int = 64,
        num_epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_split: float = 0.0,
        device: Optional[str] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.shuffle = shuffle

        self.to(self.device)

    def reset_parameters(self):
        """Reinitialize all weights of the MLP."""
        for module in self.net.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, List[float]]:
        """
        Train the decoder using mean squared error loss.

        Args:
            x: Latent features, shape (N, D_latent).
            y: Target actions, shape (N, D_action).

        Returns:
            History dict containing train/val loss curves.
        """

        from torch.utils.data import DataLoader, TensorDataset
        self.reset_parameters()

        x = x.to(self.device)
        y = y.to(self.device)

        n_samples = x.shape[0]
        if n_samples == 0:
            raise ValueError("No samples available to train decoder.")

        n_val = int(n_samples * self.val_split)
        if n_val > 0:
            idx = torch.randperm(n_samples, device=self.device)
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
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=False,
        )

        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for _ in range(self.num_epochs):
            self.train()
            total_loss = 0.0
            total_batches = 0

            for xb, yb in train_loader:
                optim.zero_grad()
                pred = self(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()

                total_loss += loss.item()
                total_batches += 1

            mean_train_loss = total_loss / max(1, total_batches)
            history["train_loss"].append(mean_train_loss)

            if n_val > 0 and x_val.numel() > 0:
                self.eval()
                with torch.no_grad():
                    val_pred = self(x_val)
                    val_loss = loss_fn(val_pred, y_val).item()
            else:
                val_loss = float("nan")
            history["val_loss"].append(val_loss)

        return history

def _pairwise_chebyshev(x: np.ndarray) -> np.ndarray:
    # x: (N, D)
    N, D = x.shape
    dist = np.zeros((N, N), dtype=np.float64)

    for d in range(D):
        col = x[:, d]
        diff = np.abs(col[:, None] - col[None, :])  # (N, N)
        np.maximum(dist, diff, out=dist)

    np.fill_diagonal(dist, 0.0)
    return dist



def _pairwise_l2(x: np.ndarray) -> np.ndarray:
    # x: (N, D)
    x = x.astype(np.float64, copy=False)
    x2 = np.sum(x * x, axis=1, keepdims=True)  # (N, 1)
    dist2 = x2 + x2.T - 2.0 * (x @ x.T)        # (N, N)
    np.maximum(dist2, 0.0, out=dist2)          # numerical safety
    np.fill_diagonal(dist2, 0.0)
    return np.sqrt(dist2, out=dist2)



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


def knn_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_samples: Optional[int] = 5000,
) -> float:
    n = min(x.shape[0], y.shape[0])
    if n == 0 or n <= k:
        return float("nan")

    x = x[:n]
    y = y[:n]

    # Optional subsampling for large n (configurable instead of fixed 10000)
    if max_samples is not None and n > max_samples:      # <-- CHANGED
        indices = rng.permutation(n)[:max_samples]
        x = x[indices]
        y = y[indices]
        n = x.shape[0]  # <-- still important

    xy = np.concatenate([x, y], axis=1)

    dist_xy = _pairwise_chebyshev(xy)
    np.fill_diagonal(dist_xy, np.inf)
    kth = np.partition(dist_xy, kth=k - 1, axis=1)[:, k - 1] + 1e-10

    dist_x = _pairwise_chebyshev(x)
    dist_y = _pairwise_chebyshev(y)
    np.fill_diagonal(dist_x, np.inf)
    np.fill_diagonal(dist_y, np.inf)

    # Count neighbors within joint-radius in the marginals
    nx = (dist_x < kth[:, None]).sum(axis=1)
    ny = (dist_y < kth[:, None]).sum(axis=1)

    # Kraskov-like estimator
    return (
        _digamma(k)
        + _digamma(n)
        - float(np.mean(_digamma_vector(nx + 1) + _digamma_vector(ny + 1)))
    )




def knn_consistency(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_samples: int,
) -> float:
    """
    k-NN label consistency for composite labels.

    Args:
        features: (N, D)
        labels:   (N,) composite action labels.
    """
    if features.shape[0] == 0 or labels.shape[0] == 0:
        return float("nan")

    if labels.ndim != 1:
        raise ValueError(
            f"knn_consistency expects composite labels with shape (N,), got {labels.shape}"
        )

    if features.shape[0] <= k:
        return float("nan")

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


def compute_class_variances(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute intra- and inter-class variance for composite labels.

    Args:
        features: (N, D)
        labels:   (N,) composite labels
    """
    if features.size == 0 or labels.size == 0:
        return float("nan"), float("nan")

    if labels.ndim != 1:
        raise ValueError(
            f"compute_class_variances expects composite labels with shape (N,), got {labels.shape}"
        )

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
    ):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.z_a_decoder: Optional[LatentToActionDecoder] = None
        self.z_q_decoder: Optional[LatentToActionDecoder] = None
        self.z_q_split_decoder: Optional[LatentToActionDecoder] = None
        self.z_a_split_decoder: Optional[LatentToActionDecoder] = None
        # full training histories per decoder
        self.decoder_histories: Dict[str, Dict[str, List[float]]] = {}
        self.num_splits = config.num_splits
        self.config.num_codes = config.num_codes

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
                z_q_t.shape[1] % self.num_splits == 0
            ), "z_q dimensionality must be divisible by num_splits"
            assert (
                act_t.shape[1] % self.num_splits == 0
            ), "Action dimensionality must be divisible by num_splits"

            self.z_q_split_decoder = LatentToActionDecoder(
                input_dim=z_q_t.shape[1] // self.num_splits,
                action_dim=act_t.shape[1] // self.num_splits,
            )
        if self.z_a_split_decoder is None:
            assert (
                z_a_t.shape[1] % self.num_splits == 0
            ), "z_a dimensionality must be divisible by num_splits"
            assert (
                act_t.shape[1] % self.num_splits == 0
            ), "Action dimensionality must be divisible by num_splits"

            self.z_a_split_decoder = LatentToActionDecoder(
                input_dim=z_a_t.shape[1] // self.num_splits,
                action_dim=act_t.shape[1] // self.num_splits,
            )

        hist_a = self.z_a_decoder.fit(z_a_t, act_t)
        hist_q = self.z_q_decoder.fit(z_q_t, act_t)

        z_q_split = z_q_t.reshape(
            -1, self.num_splits, z_q_t.shape[1] // self.num_splits
        ).reshape(-1, z_q_t.shape[1] // self.num_splits)
        z_a_split = z_a_t.reshape(
            -1, self.num_splits, z_a_t.shape[1] // self.num_splits
        ).reshape(-1, z_a_t.shape[1] // self.num_splits)
        act_split = act_t.reshape(
            -1, self.num_splits, act_t.shape[1] // self.num_splits
        ).reshape(-1, act_t.shape[1] // self.num_splits)

        hist_q_split = self.z_q_split_decoder.fit(z_q_split, act_split)
        hist_a_split = self.z_a_split_decoder.fit(z_a_split, act_split)

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

    def _predict_actions_from_split(
        self, z: np.ndarray, decoder
    ) -> np.ndarray:
        """Run a trained split decoder (z_q or z_a) to predict primitive actions."""

        if decoder is None or z.size == 0:
            return np.empty((0,))

        device = next(decoder.parameters()).device
        z_t = torch.from_numpy(z).float().to(device)

        # split into [batch, num_splits, dim_per_split] then flatten splits to feed decoder
        if z_t.shape[1] % self.num_splits != 0:
            raise ValueError(
                f"Latent dim {z_t.shape[1]} is not divisible by num_splits={self.num_splits}"
            )

        dim_per_split = z_t.shape[1] // self.num_splits
        z_split = z_t.reshape(-1, self.num_splits, dim_per_split).reshape(-1, dim_per_split)

        decoder.eval()
        with torch.no_grad():
            pred_split = decoder(z_split).cpu().numpy()

        # reshape back to [batch, num_splits * action_dim]
        return pred_split.reshape(-1, self.num_splits, pred_split.shape[1]).reshape(
            -1, pred_split.shape[1] * self.num_splits
        )

    def plot_confusion_heatmap(self, confusion: np.ndarray, title: str = "Confusion Matrix", dpi: int = 300):
        import matplotlib.pyplot as plt
        

        num_classes = confusion.shape[0]

        plt.style.use("seaborn-v0_8-white")
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=dpi)

        im = ax.imshow(confusion, cmap="Reds", aspect="auto")

        # Extract frame color for consistent grid appearance
        spine_color = ax.spines["left"].get_edgecolor()

        # ticks
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels([str(i) for i in range(num_classes)])
        ax.set_yticklabels([str(i) for i in range(num_classes)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # grid using the spine color
        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color=spine_color, linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotations
        max_val = confusion.max() if confusion.max() > 0 else 1
        for i in range(num_classes):
            for j in range(num_classes):
                val = int(confusion[i, j])
                color = "black" if val < 0.6 * max_val else "white"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14, pad=12)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Count", fontsize=11)

        fig.tight_layout()
        return fig



    def plot_per_action_accuracy(self, per_acc: np.ndarray, title="Per-Action Recall"):
        import matplotlib.pyplot as plt

        C = len(per_acc)

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

        x = np.arange(C)

        # grey bar for NaN (missing actions), blue for valid values
        colors = ["#aaaaaa" if np.isnan(a) else "#1f77b4" for a in per_acc]

        ax.bar(x, np.nan_to_num(per_acc, nan=0.0), color=colors)

        # axis labels
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(C)], rotation=45, ha="right")

        ax.set_ylabel("Recall", fontsize=12)
        ax.set_xlabel("Action", fontsize=12)
        ax.set_title(title, fontsize=14, pad=12)

        # annotate bars
        for i, val in enumerate(per_acc):
            label = "NaN" if np.isnan(val) else f"{val:.2f}"
            ax.text(i, (0 if np.isnan(val) else val) + 0.02,
                    label, ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        return fig


    def _compute_confusion_matrix(
        self, true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_classes = int(
            max(
                self.centroids.shape[0],
                (true_labels.max() + 1) if true_labels.size else 0,
                (pred_labels.max() + 1) if pred_labels.size else 0,
            )
        )
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true_labels.tolist(), pred_labels.tolist()):
            confusion[int(t), int(p)] += 1

        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion, dtype=float),
            where=row_sums > 0,
        )
        return confusion, normalized

    def _compute_macro_f1(self, confusion: np.ndarray) -> float:
        tp = np.diag(confusion).astype(float)
        fp = confusion.sum(axis=0) - tp
        fn = confusion.sum(axis=1) - tp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        return float(np.mean(f1))

    def compute_per_action_recall(self, confusion: np.ndarray):
        """
        Returns per-action recall with NaN for classes that never appear.
        """
        tp = np.diag(confusion).astype(float)
        fn = confusion.sum(axis=1) - tp
        denom = tp + fn

        recall = np.divide(tp, denom, out=np.full_like(tp, np.nan, dtype=float), where=(denom > 0))
        return recall, np.nanmean(recall)

    def _compute_split_decoder_classification_metrics(
        self,
        z: np.ndarray,
        actions: np.ndarray,
        decoder,
        prefix: str,
    ) -> Dict[str, object]:
        """
        Compute classification metrics for a split decoder (z_q or z_a).

        prefix examples:
        - "z_q_split"
        - "z_a_split"
        """
        if z.size == 0 or actions.size == 0:
            return {}

        pred_actions = self._predict_actions_from_split(z, decoder)
        if pred_actions.size == 0:
            return {}

        true_labels = self._actions_to_labels(actions).reshape(-1)
        pred_labels = self._actions_to_labels(pred_actions).reshape(-1)

        # confusion matrix and macro-F1
        confusion, _ = self._compute_confusion_matrix(true_labels, pred_labels)
        macro_f1 = self._compute_macro_f1(confusion)

        # figures
        confusion_fig = self.plot_confusion_heatmap(
            confusion,
            f"Decoder ({prefix}): Codes Action Confusion Matrix",
        )
        per_action_acc, _ = self.compute_per_action_recall(confusion)
        per_action_acc_fig = self.plot_per_action_accuracy(
            per_action_acc,
            f"Decoder ({prefix}): Codes Action Accuracy",
        )

        return {
            f"{prefix}_macro_f1": macro_f1,
            f"{prefix}_confusion_heatmap_fig": confusion_fig,
            f"{prefix}_per_action_accuracy_fig": per_action_acc_fig,
        }




    def reset(self) -> None:
        self.latent_actions: List[np.ndarray] = []
        self.quantized_actions: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.state_curr: List[np.ndarray] = []
        self.state_next: List[np.ndarray] = []
        self.primitive_labels: List[np.ndarray] = []
        self.composite_labels: List[np.ndarray] = []
        self.code_indices: List[np.ndarray] = []
        self.pred_errors: List[np.ndarray] = []
        self.pred_error_labels: List[np.ndarray] = []

    def _collapse_labels(self, labels: np.ndarray) -> np.ndarray:
        """Reduce per-split labels to a single composite label via majority vote."""

        composite: List[int] = []
        for row in labels:
            vals, counts = np.unique(row, return_counts=True)
            maxc = counts.max()
            majority = vals[counts == maxc]
            if majority.size == 1:
                composite.append(int(majority[0]))
            else:
                composite.append(int(row[0]))
        return np.array(composite, dtype=int)

    def _actions_to_labels(self, actions: np.ndarray) -> np.ndarray:
        """
        Convert stacked 2-D centroid actions into their centroid indices.

        Input:
            actions: (N, num_splits * 2)
        Output:
            labels:  (N, num_splits) with integer centroid IDs in {0..8}
        """
        base_action_dim = 2
        expected_dim = self.num_splits * base_action_dim

        if actions.ndim != 2:
            raise ValueError(f"actions must be 2D, got shape {actions.shape}")
        if actions.shape[1] != expected_dim:
            raise ValueError(
                f"actions second dim must be num_splits * 2 = {expected_dim}, "
                f"got {actions.shape[1]}"
            )

        actions = actions.reshape(-1, self.num_splits, base_action_dim)
        N = actions.shape[0]
        labels = np.zeros((N, self.num_splits), dtype=np.int64)

        for i in range(self.num_splits):
            sub = actions[:, i, :]  # (N, 2)
            diff = sub[:, None, :] - self.centroids[None, :, :]  # (N, 9, 2)
            dist = np.sum(diff * diff, axis=2)  # (N, 9)
            labels[:, i] = np.argmin(dist, axis=1)

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
            codes = code_indices[:, :valid_steps]   # (B, valid_steps, self.num_splits)

            # Sanity checks for num_splits consistency
            if codes.shape[-1] != self.num_splits:
                raise ValueError(
                    "[LatentMetricsAggregator.update] "
                    f"code_indices.shape={code_indices.shape}, "
                    f"codes.shape={codes.shape}, "
                    f"num_splits={self.num_splits}"
                )

            expected_action_dim = 2 * self.num_splits
            if act_curr.shape[1] != expected_action_dim:
                raise ValueError(
                    "[LatentMetricsAggregator.update] "
                    f"actions (torch) shape={actions.shape}, "
                    f"act_curr (np) shape={act_curr.shape}, "
                    f"num_splits={self.num_splits}, "
                    f"expected_action_dim={expected_action_dim}"
                )

                          
            labels = self._actions_to_labels(act_curr)           # (M, self.num_splits)
            composite_labels = self._collapse_labels(labels)
            primitive_labels = labels
            codes_flat = codes.reshape(-1).cpu().numpy()          # (M * self.num_splits,)                           # (M * self.num_splits,)


            self.latent_actions.append(z_a)
            self.quantized_actions.append(z_q)
            self.actions.append(act_curr)
            self.state_curr.append(state_t)
            self.state_next.append(state_tp1)
            self.composite_labels.append(composite_labels)
            self.primitive_labels.append(primitive_labels)
            self.code_indices.append(codes_flat)

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
        if codes.size == 0:
            return {
                "code_perplexity": float("nan"),
                "dead_code_fraction": float("nan"),
                "bits_per_action": float("nan"),
            }

        if codes.min() < 0:
            raise ValueError(f"codes must be non-negative, got min={codes.min()}")
        if codes.max() >= self.config.num_codes:
            raise ValueError(
                f"Encountered code index {codes.max()} >= num_codes={self.config.num_codes}"
            )

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
        import matplotlib.pyplot as plt
        from collections import defaultdict

        if codes.size == 0 or labels.size == 0:
            return None

        # Collect codes per action
        per_action_codes = defaultdict(set)
        for code, label in zip(codes.tolist(), labels.tolist()):
            per_action_codes[int(label)].add(int(code))

        # --- FIX: Always use 9 actions (0–8) ---
        actions = list(range(9))  # <-- ONLY CHANGE
        A = len(actions)

        # Build Jaccard matrix
        overlap_matrix = np.zeros((A, A), dtype=float)
        for i, ai in enumerate(actions):
            Ci = per_action_codes.get(ai, set())
            for j, aj in enumerate(actions):
                Cj = per_action_codes.get(aj, set())
                union = len(Ci | Cj)
                inter = len(Ci & Cj)
                overlap_matrix[i, j] = (inter / union) if union > 0 else 0.0

        # Plotting
        plt.style.use("seaborn-v0_8-white")
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

        im = ax.imshow(overlap_matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(np.arange(A))
        ax.set_yticks(np.arange(A))
        ax.set_xticklabels(actions)
        ax.set_yticklabels(actions)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Use spine color for grid
        spine_color = ax.spines["left"].get_edgecolor()

        ax.set_xticks(np.arange(-0.5, A, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, A, 1), minor=True)
        ax.grid(which="minor", color=spine_color, linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotate
        for i in range(A):
            for j in range(A):
                val = overlap_matrix[i, j]
                color = "black" if val < 0.55 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

        ax.set_xlabel("Action", fontsize=12)
        ax.set_ylabel("Action", fontsize=12)
        ax.set_title("Action–Action Code Usage Overlap", fontsize=14, pad=12)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Similarity", fontsize=11)

        fig.tight_layout()
        return fig
    
    def _compute_umap_double(
        self,
        z_a: np.ndarray,
        z_q: np.ndarray,
        composite_labels: np.ndarray,
        primitive_labels: Optional[np.ndarray] = None,
    ):
        if (
            z_a.size == 0
            or z_q.size == 0
            or composite_labels.size == 0
        ):
            return None, None, None

        assert composite_labels.ndim == 1

        N_full = z_a.shape[0]
        N = min(N_full, self.config.umap_points)
        idx = (
            np.arange(N_full)
            if N_full <= N
            else self.rng.permutation(N_full)[:N]
        )

        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        import umap
        import numpy as np

        z_a_s = StandardScaler().fit_transform(z_a[idx])
        z_q_s = StandardScaler().fit_transform(z_q[idx])
        labels = composite_labels[idx]

        plt.style.use("seaborn-v0_8-whitegrid")

        def make_reducer():
            return umap.UMAP(
                n_neighbors=self.config.umap_neighbors,
                min_dist=self.config.umap_min_dist,
                random_state=self.config.seed,
                metric="euclidean",
            )

        reducer_a = make_reducer()
        reducer_q = make_reducer()

        emb_a = reducer_a.fit_transform(z_a_s)
        emb_q = reducer_q.fit_transform(z_q_s)

        cmap = plt.get_cmap("tab20")

        def nice_umap(emb: np.ndarray, labels_local: np.ndarray, title: str):
            fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
            ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=labels_local,
                cmap=cmap,
                s=12,
                alpha=0.7,
                linewidth=0,
                edgecolors="none",
            )
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
            fig.tight_layout()
            return fig

        # Composite-label UMAPs
        fig_a = nice_umap(emb_a, labels, "UMAP z_a (composite)")
        fig_q = nice_umap(emb_q, labels, "UMAP z_q (composite)")

        # ----------------------------------------------------------
        # z_a split UMAP with primitive labels (optional)
        # ----------------------------------------------------------
        fig_a_prim = None
        if primitive_labels is not None:
            primitive_labels = np.asarray(primitive_labels)
            if (
                primitive_labels.ndim != 2
                or primitive_labels.shape[0] != N_full
                or primitive_labels.shape[1] != self.num_splits
            ):
                raise ValueError(
                    f"primitive_labels must have shape (N_full, num_splits) = "
                    f"({N_full}, {self.num_splits}), got {primitive_labels.shape}"
                )

            # Subsample + flatten per split
            prim_labels_s = primitive_labels[idx].reshape(-1)

            D = z_a.shape[1]
            if D % self.num_splits != 0:
                raise ValueError(
                    f"z_a dim {D} not divisible by num_splits={self.num_splits}"
                )
            dim_per_split = D // self.num_splits

            # (N, D) -> (N, S, D/S) -> (N*S, D/S)
            z_a_split = z_a[idx].reshape(N, self.num_splits, dim_per_split).reshape(
                -1, dim_per_split
            )
            z_a_split_s = StandardScaler().fit_transform(z_a_split)

            reducer_split = make_reducer()
            emb_a_split = reducer_split.fit_transform(z_a_split_s)

            fig_a_prim = nice_umap(
                emb_a_split,
                prim_labels_s,
                "UMAP z_a splits (primitive)",
            )

        return fig_a, fig_q, fig_a_prim



    def _build_confusion_heatmap(self, codes: np.ndarray, labels: np.ndarray):
        import matplotlib.pyplot as plt

        if codes.size == 0 or labels.size == 0:
            return None

        # ----------------------------------------
        # FIXED ACTION RANGE: always 9 actions (0..8)
        # ----------------------------------------
        num_actions = 9
        num_codes = self.config.num_codes

        # Build confusion matrix (codes × actions)
        confusion = np.zeros((num_codes, num_actions), dtype=float)

        for code, label in zip(codes.tolist(), labels.tolist()):
            if 0 <= code < num_codes and 0 <= label < num_actions:
                confusion[code, label] += 1

        # ----------------------------------------
        # Plotting (high DPI + frame-colored grid)
        # ----------------------------------------
        plt.style.use("seaborn-v0_8-white")
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

        im = ax.imshow(confusion, cmap="Reds", aspect="auto")

        # Ticks
        ax.set_xticks(np.arange(num_actions))
        ax.set_yticks(np.arange(num_codes))
        ax.set_xticklabels([str(i) for i in range(num_actions)])
        ax.set_yticklabels([str(i) for i in range(num_codes)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Use frame (spine) color for grid
        spine_color = ax.spines["left"].get_edgecolor()

        # Grid on top of the heatmap
        ax.set_xticks(np.arange(-0.5, num_actions, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_codes, 1), minor=True)
        ax.grid(which="minor", color=spine_color, linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotate counts
        max_val = confusion.max() if confusion.max() > 0 else 1
        threshold = 0.6 * max_val

        for i in range(num_codes):
            for j in range(num_actions):
                val = int(confusion[i, j])
                color = "white" if confusion[i, j] > threshold else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=9, color=color)

        # Labels
        ax.set_xlabel("Finegrained Action", fontsize=12)
        ax.set_ylabel("Code Index", fontsize=12)
        ax.set_title("Code–Action Confusion Matrix", fontsize=14, pad=12)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("Count", fontsize=11)

        fig.tight_layout()
        return fig




    def compute(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        metrics: Dict[str, float] = {}
        figures: Dict[str, object] = {}

        if not self.latent_actions:
            return metrics, figures

        z_a = self._stack(self.latent_actions)
        z_q = self._stack(self.quantized_actions)
        actions = self._stack(self.actions)
        state_t = self._stack(self.state_curr)
        state_tp1 = self._stack(self.state_next)
        composite_labels = self._stack(self.composite_labels)    # (N,)
        primitive_labels = self._stack(self.primitive_labels)    # (N, num_splits)
        codes = self._stack(self.code_indices).astype(int)       # (N * num_splits,)


        # Sanity: primitive labels shape
        if primitive_labels.size > 0 and primitive_labels.ndim != 2:
            raise ValueError(
                f"primitive_labels expected shape (N, num_splits), got {primitive_labels.shape}"
            )

        # --- Code usage (no labels) ---
        metrics.update(self._compute_code_usage(codes))

        # --- Class variances & FDR: use composite labels ---
        intra_a, inter_a = compute_class_variances(z_a, composite_labels)
        intra_q, inter_q = compute_class_variances(z_q, composite_labels)
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

        # --- kNN consistency: use composite labels ---
        metrics.update(
            {
                "z_a_knn_consistency": knn_consistency(
                    z_a, composite_labels, self.config.knn_k, self.rng, self.config.max_samples
                ),
                "z_q_knn_consistency": knn_consistency(
                    z_q, composite_labels, self.config.knn_k, self.rng, self.config.max_samples
                ),
            }
        )

        # --- Mutual information: uses continuous actions/states directly ---
        metrics.update(
            {
                "mi_z_a_action": knn_mutual_information(z_a, actions, self.config.mi_k, self.rng),
                "mi_z_a_state_t": knn_mutual_information(
                    z_a, state_t, self.config.mi_k, self.rng
                ),
                "mi_z_a_state_tp1": knn_mutual_information(
                    z_a, state_tp1, self.config.mi_k, self.rng
                ),
                "mi_z_q_action": knn_mutual_information(z_q, actions, self.config.mi_k, self.rng),
                "mi_z_q_state_t": knn_mutual_information(
                    z_q, state_t, self.config.mi_k, self.rng
                ),
                "mi_z_q_state_tp1": knn_mutual_information(
                    z_q, state_tp1, self.config.mi_k, self.rng
                ),
            }
        )

        # --- Primitive labels flattened for code–action metrics ---
        primitive_labels_flat = (
            primitive_labels.reshape(-1) if primitive_labels.size > 0 else primitive_labels
        )

        # --- Code–primitive-action confusion heatmap ---
        confusion_fig = self._build_confusion_heatmap(codes, primitive_labels_flat)
        if confusion_fig is not None:
            figures["code_primitive_action_confusion"] = confusion_fig

        # --- Primitive-action ↔ primitive-action overlap (Jaccard) ---
        overlap_fig = self._compute_action_action_overlap(codes, primitive_labels_flat)
        if overlap_fig is not None:
            figures["primitive_action_overlap"] = overlap_fig

        # --- UMAP : composite labels only (z_a and z_q) + z_a splits with primitive labels ---
        umap_fig_a, umap_fig_q, umap_fig_a_prim = self._compute_umap_double(
            z_a,
            z_q,
            composite_labels,
            primitive_labels=primitive_labels,  # or primitive_labels
        )

        if umap_fig_a is not None:
            figures["umap_z_a"] = umap_fig_a
        if umap_fig_q is not None:
            figures["umap_z_q"] = umap_fig_q
        if umap_fig_a_prim is not None:
            figures["umap_z_a_split_primitive"] = umap_fig_a_prim


        # --- Train latent->action decoders on accumulated data ---
        decoder_metrics = self._train_action_decoders(z_a, z_q, actions)
        metrics.update(decoder_metrics)

        # --- Split decoder classification metrics (returns macroF1 and 2 figures) ---
        split_metrics = {}
        # z_q metrics
        split_metrics.update(
            self._compute_split_decoder_classification_metrics(
                z=z_q,
                actions=actions,
                decoder=self.z_q_split_decoder,
                prefix="z_q_split",
            )
        )
        # z_a metrics
        split_metrics.update(
            self._compute_split_decoder_classification_metrics(
                z=z_a,
                actions=actions,
                decoder=self.z_a_split_decoder,
                prefix="z_a_split",
            )
        )
        # z_q
        if "z_q_split_confusion_heatmap_fig" in split_metrics:
            figures["z_q_split_confusion_heatmap"] = split_metrics.pop(
                "z_q_split_confusion_heatmap_fig"
            )

        if "z_q_split_per_action_accuracy_fig" in split_metrics:
            figures["z_q_split_per_action_accuracy"] = split_metrics.pop(
                "z_q_split_per_action_accuracy_fig"
            )

        # z_a
        if "z_a_split_confusion_heatmap_fig" in split_metrics:
            figures["z_a_split_confusion_heatmap"] = split_metrics.pop(
                "z_a_split_confusion_heatmap_fig"
            )

        if "z_a_split_per_action_accuracy_fig" in split_metrics:
            figures["z_a_split_per_action_accuracy"] = split_metrics.pop(
                "z_a_split_per_action_accuracy_fig"
            )

        # Keep macroF1 in metrics
        metrics.update(split_metrics)

        return metrics, figures


