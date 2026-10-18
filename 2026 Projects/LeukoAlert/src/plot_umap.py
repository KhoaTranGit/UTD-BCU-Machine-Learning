
# !pip install umap-learn
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def plot_umap(
    X,
    y=None,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    scale=True,
    random_state=42,
    figsize=(9, 7),
    title="UMAP Projection"
):
    """
    Visualize high-dimensional data using 2D UMAP.

    Parameters
    ----------
    X : pandas.DataFrame, polars.DataFrame, or np.ndarray
        Numeric feature matrix.
    y : array-like, optional
        Class labels used to color observations.
    n_neighbors : int, default=15
        Number of neighbors used to estimate local structure.
    min_dist : float, default=0.1
        Minimum distance between points in the embedding.
    metric : str, default="euclidean"
        Distance metric used by UMAP.
    scale : bool, default=True
        Whether to standardize features before UMAP.
    random_state : int, default=42
        Random seed for reproducibility.
    figsize : tuple, default=(9, 7)
        Figure dimensions.
    title : str, default="UMAP Projection"
        Plot title.

    Returns
    -------
    np.ndarray
        Two-dimensional UMAP embedding.
    """
    # Convert to NumPy (supports Polars and pandas)
    X = np.asarray(X, dtype=float)
    # Handle missing values
    X = SimpleImputer(strategy="median").fit_transform(X)
    # Standardize features
    if scale:
      X = StandardScaler().fit_transform(X)

    # Compute UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(X)

    # Visualize
    fig, ax = plt.subplots(figsize=figsize)

    if y is None:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=0.65,
            s=15
        )
    else:
        y = np.asarray(y).ravel()
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows.")

        for label in np.unique(y):
            mask = y == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=str(label),
                alpha=0.65,
                s=15
            )

        ax.legend(title="Class")

    ax.set(
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title=title
    )
    plt.tight_layout()
    plt.show()

    return embedding
