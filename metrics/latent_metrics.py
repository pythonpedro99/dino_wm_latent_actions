import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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


def knn_consistency(features: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator, max_samples: int) -> float:
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



def compute_class_variances(features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if features.size == 0:
        return float("nan"), float("nan")
    unique_labels = np.unique(labels)
    class_means = []
    intra = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue
        class_feats = features[mask]
        intra.append(class_feats.var(axis=0, ddof=1).mean())
        class_means.append(class_feats.mean(axis=0))

    if not class_means:
        return float("nan"), float("nan")

    class_means = np.stack(class_means)
    intra_val = float(np.mean(intra)) if intra else float("nan")
    inter_dists = _pairwise_l2(class_means)
    triu = np.triu_indices_from(inter_dists, k=1)
    if triu[0].size == 0:
        inter_val = float("nan")
    else:
        inter_val = float(np.mean(inter_dists[triu] ** 2))
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


class LatentMetricsAggregator:
    def __init__(self, config: LatentMetricsConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
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

            # Flatten codes directly; frameskip == num_splits
            codes_flat = codes.reshape(-1).cpu().numpy()          # (M * frameskip,)

            # Labels already match codes; no repetition needed
            code_labels = labels


            self.latent_actions.append(z_a)
            self.quantized_actions.append(z_q)
            self.actions.append(act_curr)
            self.state_curr.append(state_t)
            self.state_next.append(state_tp1)
            self.labels.append(labels)
            self.code_indices.append(codes_flat)
            self.code_labels.append(code_labels)

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


    def _compute_umap(self, features: np.ndarray, labels: np.ndarray, prefix: str) -> Optional["matplotlib.figure.Figure"]:
        if features.shape[0] == 0:
            return None
        try:
            import matplotlib.pyplot as plt
            import umap
        except Exception:
            return None

        sample_features, indices = _sample_with_indices(
            features, self.config.umap_points, self.rng
        )
        sample_labels = labels[indices]
        reducer = umap.UMAP(
            n_neighbors=self.config.umap_neighbors,
            min_dist=self.config.umap_min_dist,
            random_state=self.config.seed,
        )
        embedding = reducer.fit_transform(sample_features)
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=sample_labels, cmap="tab20", s=6, alpha=0.8
        )
        ax.set_title(f"UMAP {prefix}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def _compute_per_action_errors(self) -> Tuple[float, Dict[int, float]]:
        if not self.pred_errors:
            return float("nan"), {}
        errors = self._stack(self.pred_errors)
        labels = self._stack(self.pred_error_labels)
        overall = float(np.mean(errors)) if errors.size else float("nan")
        per_action = {}
        for label in np.unique(labels):
            mask = labels == label
            per_action[int(label)] = float(errors[mask].mean())
        return overall, per_action

    def _plot_heatmap(
        self,
        matrix: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str,
        cbar_label: str,
        cmap: str = "magma",
    ) -> Optional["matplotlib.figure.Figure"]:
        if matrix.size == 0:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Action")
        ax.set_ylabel("Code")
        ax.set_title(title)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        fig.tight_layout()
        return fig

    def _plot_bar_chart(
        self, values: Dict[int, float], title: str, ylabel: str
    ) -> Optional["matplotlib.figure.Figure"]:
        if not values:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        actions = sorted(values.keys())
        data = [values[a] for a in actions]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(actions, data, color="#4c72b0")
        ax.set_xlabel("Action")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(actions)
        fig.tight_layout()
        return fig

    def _build_confusion_heatmap(
        self, codes: np.ndarray, labels: np.ndarray
    ) -> Tuple[Optional["matplotlib.figure.Figure"], Dict[str, float]]:
        if codes.size == 0 or labels.size == 0:
            return None, {
                "code_label_entropy_mean": float("nan"),
                "code_label_entropy_min": float("nan"),
                "code_label_entropy_max": float("nan"),
            }

        num_actions = int(labels.max()) + 1 if labels.size else 0
        confusion = np.zeros((self.config.num_codes, num_actions), dtype=float)
        per_code_entropies = []

        for code, label in zip(codes.tolist(), labels.tolist()):
            if code < self.config.num_codes and label < num_actions:
                confusion[code, label] += 1

        for row in confusion:
            total = row.sum()
            if total == 0:
                per_code_entropies.append(float("nan"))
                continue
            probs = row / total
            mask = probs > 0
            entropy = -np.sum(probs[mask] * np.log2(probs[mask])) if mask.any() else float("nan")
            per_code_entropies.append(float(entropy))

        entropy_array = np.array([v for v in per_code_entropies if not math.isnan(v)])
        stats = {
            "code_label_entropy_mean": float(entropy_array.mean()) if entropy_array.size else float("nan"),
            "code_label_entropy_min": float(entropy_array.min()) if entropy_array.size else float("nan"),
            "code_label_entropy_max": float(entropy_array.max()) if entropy_array.size else float("nan"),
        }

        x_labels = [str(i) for i in range(num_actions)]
        y_labels = [str(i) for i in range(self.config.num_codes)]
        fig = self._plot_heatmap(
            confusion, x_labels, y_labels, "Code-Action Confusion", "Count"
        )
        return fig, stats

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

        umap_a = self._compute_umap(z_a, labels, prefix="z_a")
        umap_q = self._compute_umap(z_q, labels, prefix="z_q")
        if umap_a is not None:
            figures["umap_z_a"] = umap_a
        if umap_q is not None:
            figures["umap_z_q"] = umap_q

        return metrics, tables, figures
