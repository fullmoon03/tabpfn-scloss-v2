from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mean_belief(
    mean_belief: np.ndarray,
    std_belief: np.ndarray,
    query_indices: np.ndarray,
    class_labels: np.ndarray,
    queries_per_figure: int,
    plot_rows: int,
    plot_cols: int,
    outdir: str,
    dataset_name: str,
) -> None:
    depth = np.arange(mean_belief.shape[0])
    num_queries = mean_belief.shape[1]
    num_classes = mean_belief.shape[2]
    class_handles = None

    for fig_idx, start in enumerate(range(0, num_queries, queries_per_figure), start=1):
        end = min(start + queries_per_figure, num_queries)
        fig, axes = plt.subplots(
            plot_rows,
            plot_cols,
            figsize=(plot_cols * 4.0, plot_rows * 3.0),
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(-1)

        for ax_idx, query_pos in enumerate(range(start, end)):
            ax = axes[ax_idx]
            line_handles = []
            for class_idx in range(num_classes):
                (line,) = ax.plot(
                    depth,
                    mean_belief[:, query_pos, class_idx],
                    linewidth=1.5,
                    marker="o",
                    markersize=2.2,
                    label=f"class {class_idx} ({class_labels[class_idx]})",
                )
                lower = np.clip(
                    mean_belief[:, query_pos, class_idx] - std_belief[:, query_pos, class_idx],
                    0.0,
                    1.0,
                )
                upper = np.clip(
                    mean_belief[:, query_pos, class_idx] + std_belief[:, query_pos, class_idx],
                    0.0,
                    1.0,
                )
                ax.fill_between(depth, lower, upper, color=line.get_color(), alpha=0.18)
                line_handles.append(line)
            if class_handles is None:
                class_handles = line_handles

            ax.set_title(f"q_idx={query_indices[query_pos]}")
            ax.set_xlim(depth[0], depth[-1])
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)

        for ax in axes[end - start :]:
            ax.axis("off")

        fig.suptitle(
            f"{dataset_name.capitalize()} fixed-query mean belief across paths "
            f"({start + 1}-{end}/{num_queries})",
            y=0.995,
        )
        fig.supxlabel("Depth n")
        fig.supylabel("Mean Belief")
        if class_handles is not None:
            fig.legend(
                class_handles,
                [handle.get_label() for handle in class_handles],
                loc="upper right",
                frameon=True,
            )
        fig.tight_layout(rect=(0, 0, 0.97, 0.97))
        fig.savefig(
            f"{outdir}/belief-trajectory-{fig_idx}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
