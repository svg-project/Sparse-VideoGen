import numpy as np
import pytest
import torch
from cuvs.cluster.kmeans import KMeansParams, cluster_cost, fit, predict
from cuvs.distance import pairwise_distance


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [64, 128])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("hierarchical", [False])
def test_kmeans_fit_torch(n_rows, n_cols, n_clusters, dtype, hierarchical):
    if hierarchical and dtype == torch.float64:
        pytest.skip("hierarchical kmeans doesn't support float64")

    # generate some random input points / centroids
    X_host = torch.rand((n_rows, n_cols), dtype=dtype)
    centroids = X_host[:n_clusters].clone().cuda()
    X = X_host.clone().cuda()

    # compute the inertia, before fitting centroids
    original_inertia = cluster_cost(X, centroids)

    params = KMeansParams(n_clusters=n_clusters, hierarchical=hierarchical)

    # fit the centroids, make sure inertia has gone down
    centroids, inertia, n_iter = fit(params, X, centroids)
    assert n_iter >= 1

    inertia = torch.tensor(inertia, device=X.device)

    # balanced kmeans doesn't return inertia
    if not hierarchical:
        assert inertia < original_inertia
        cost = cluster_cost(X, centroids)
        cost = torch.tensor(cost, device=X.device)
        assert torch.allclose(cost, inertia, rtol=1e-6)

    # make sure the prediction for each centroid is the centroid itself
    labels, inertia = predict(params, centroids, centroids)
    labels = torch.tensor(labels, device=X.device)
    assert torch.all(labels == torch.arange(labels.shape[0], device=labels.device))
