
# !pip install umap-learn
import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import OneHotEncoder

def plot_tree_umap(
    model,
    X,
    y=None,
    n_neighbors=30,
    min_dist=0.1,
    random_state=42,
    figsize=(9, 7),
    title="UMAP of Learned Tree Representation"
):
    """
    Visualize a trained tree model's learned representation
    using one-hot leaf assignments and UMAP.

    Parameters
    ----------
    model : fitted tree-based estimator
        Model supporting apply() or LightGBM's pred_leaf.
    X : array-like
        Input observations with the same features used in training.
    y : array-like, optional
        Labels used to color the embedding.
    n_neighbors : int, default=30
        UMAP neighborhood size.
    min_dist : float, default=0.1
        UMAP minimum embedding distance.
    random_state : int, default=42
        Random seed.
    figsize : tuple, default=(9, 7)
        Plot dimensions.
    title : str
        Visualization title.

    Returns
    -------
    np.ndarray
        Two-dimensional UMAP coordinates.
    """

    # Extract leaf assignments
    if model.__class__.__module__.startswith("lightgbm"):
    leaves = model.predict(X, pred_leaf=True)

    elif model.__class__.__module__.startswith("catboost"):
        leaves = model.calc_leaf_indexes(X)
    
    else:
        leaves = model.apply(X)

    leaves = np.asarray(leaves).reshape(len(X), -1)

    # One-hot encode leaf indices per tree
    encoder = OneHotEncoder(handle_unknown="ignore")
    leaf_features = encoder.fit_transform(leaves)

    # Project learned representation to 2D
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state
    )

    embedding = reducer.fit_transform(leaf_features)

    # Visualize
    fig, ax = plt.subplots(figsize=figsize)

    if y is None:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=15,
            alpha=0.65
        )
    else:
        y = np.asarray(y).ravel()

        if len(y) != len(X):
            raise ValueError("X and y must have matching row counts.")

        for label in np.unique(y):
            mask = y == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=15,
                alpha=0.65,
                label=str(label)
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
