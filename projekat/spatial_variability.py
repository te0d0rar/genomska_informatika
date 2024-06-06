import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm


def calc_entropy(x):
    """Calculates the entropy of an array, which it first normalizes.

    Args:
        x (array-like): Input array.

    Returns:
        float: Entropy of the input array.
    """

    prob_x = x.flatten()
    _, counts = np.unique(prob_x, return_counts=True)
    probs = counts / len(prob_x)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def calc_gene_entropies(adata, num_bins=10):
    """Calculates the entropy for every gene in the spatial variance gene expression.

    Args:
        adata (AnnData): Anndata object containing the data.
        num_bins (int, optional): Number of bins to use for the spatial coordinates. Defaults to 10.

    Returns:
        array-like: Entropy for every gene.
    """

    coordinates = adata.obsm["spatial"]
    num_genes = adata.shape[1]
    gene_entropy = []

    for gene_index in tqdm(range(num_genes)):
        gene_expression = adata[:, gene_index].X.toarray().flatten()
        bins = np.histogram2d(
            coordinates[:, 0], coordinates[:, 1], bins=num_bins, weights=gene_expression
        )[0]

        gene_entropy.append(calc_entropy(bins))

    return np.array(gene_entropy)


def svg(gene_entropies, percentile=90):
    """Identifies spatially variable genes from the gene entropy.

    Args:
        gene_entropies (array-like): Entropy for every gene.
        percentile (int, optional): Percentile value to use for identifying spatially variable genes. Defaults to 90.

    Returns:
        array-like: Whether a gene is spatially variable, 1 if it is, 0 otherwise.
    """

    percentile_value = np.percentile(gene_entropies, percentile)
    percentile_svg = np.array(
        [1 if x > percentile_value else 0 for x in gene_entropies]
    )

    return percentile_svg


def compute_neighbors_and_weights(coordinates, neighbors=77):
    """Computes neighbors and weights for all points based on spatial coordinates.

    Args:
        coordinates (array-like): Spatial coordinates of cells.
        neighbors (int, optional): Number of neighbors to use for the kNN regression. Defaults to 77.

    Returns:
        tuple: distances, indices, weights for all points.
    """

    knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(coordinates)

    distances, indices = knn.kneighbors(coordinates)

    distances[:, 0] = distances[:, 1]

    weights = 1 / (distances + 1e-8)  # Avoid division by zero
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights

    return indices, weights


def calc_non_uniformity(gene_expression, indices, weights):
    """Predicts gene expression using precomputed neighbors and weights.

    Args:
        gene_expression (array-like): Expression levels of a single gene.
        indices (array-like): Indices of the nearest neighbors for all points.
        weights (array-like): Weights of the nearest neighbors for all points.

    Returns:
        float: Mean of absolute differences between the predicted expression and the average expression.
    """
    neighbor_expressions = gene_expression[indices]

    # Predict expressions by weighted sum of neighbors
    predicted_expression = np.sum(weights * neighbor_expressions, axis=1)

    average_expression = np.mean(predicted_expression)

    return np.mean(np.abs(predicted_expression - average_expression))


def calc_gene_knn(adata, neighbors=77):
    """Calculates the spatial variability scores for every gene. It uses kNN to calculate the approximation and then finds the difference between the predicted
    and the average expression.

    Args:
        adata (AnnData): Anndata object containing the data.
        neighbors (int, optional): Number of neighbors to use for the kNN regression. Defaults to 77.

    Returns:
        array-like: Spatial variability scores for every gene.
    """

    coordinates = adata.obsm["spatial"]
    num_genes = adata.shape[1]

    # Compute neighbors and weights once
    indices, weights = compute_neighbors_and_weights(coordinates, neighbors)

    gene_spatial_variability = []

    for gene_index in tqdm(range(num_genes)):
        gene_expression = adata[:, gene_index].X.toarray().flatten()
        non_uniformity = calc_non_uniformity(gene_expression, indices, weights)
        gene_spatial_variability.append(non_uniformity)

    return np.array(gene_spatial_variability)
