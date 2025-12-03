
"""
python cluster.py --m resnet50 --d coco --nc 320
python cluster.py --m allroberta --d coco --nc 320 -t
"""
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

import torch
from tqdm import tqdm

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--m",
        dest="model_name",
        help="model_name",
        default="dinov2",
        type=str,
    )
    parser.add_argument(
        "--d",
        dest="dataset",
        help="dataset",
        default="coco",
        type=str,
    )
    parser.add_argument(
        "-t",
        dest="text",
        help="text",
        action='store_true',
    )
    parser.add_argument(
        "--nc",
        dest="n_clusters",
        help="number of clusters",
        default=320,
        type=int,
    )
    return parser.parse_args()

def _extract_tensor_from_loaded(obj):
    """Return the first tensor found inside obj (tensor, dict, list, tuple), or None."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        # common keys authors use
        for k in ('text_embeddings', 'text_representations', 'text_embeds',
                  'embeddings', 'image_representations', 'image_embeddings', 'emb'):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        # fall back: search values recursively
        for v in obj.values():
            t = _extract_tensor_from_loaded(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _extract_tensor_from_loaded(v)
            if t is not None:
                return t
    return None

def cluster_text(model, dataset, n_clusters):
    source_base_large = torch.load(f"data/{dataset}_{model}_text.pt")
    tensor = _extract_tensor_from_loaded(source_base_large)
    if tensor is None:
        raise RuntimeError(f"No tensor found inside loaded object from data/{dataset}_{model}_text.pt. "
                        "Expected a dict containing a tensor under 'text_embeddings' or similar.")
    # Now clustering_matrix is a numpy array on CPU
    clustering_matrix = tensor.cpu().numpy()
    kmeans = KMeans(n_clusters= n_clusters, random_state=0).fit(clustering_matrix)
    
    cos_sims = cosine_similarity(kmeans.cluster_centers_, clustering_matrix)
    closest_indices = np.argsort(cos_sims, axis=1)[:,-1:]
    closest_indices = list(set(closest_indices.reshape(-1)))
    choose = closest_indices
    
    torch.save(choose, f'data/{dataset}_{model}_text_cluster.pt')

def cluster_img(model, dataset, n_clusters):
    source_base_large = torch.load(f"data/{dataset}_{model}_img.pt")
    tensor = _extract_tensor_from_loaded(source_base_large)
    if tensor is None:
        raise RuntimeError(f"No tensor found inside loaded object from data/{dataset}_{model}_text.pt. "
                        "Expected a dict containing a tensor under 'text_embeddings' or similar.")
    # Now clustering_matrix is a numpy array on CPU
    clustering_matrix = tensor.cpu().numpy()
    kmeans = KMeans(n_clusters= n_clusters, random_state=0).fit(clustering_matrix)
    
    cos_sims = cosine_similarity(kmeans.cluster_centers_, clustering_matrix)
    closest_indices = np.argsort(cos_sims, axis=1)[:,-1:]
    closest_indices = list(set(closest_indices.reshape(-1)))
    choose = closest_indices
    
    torch.save(choose, f'data/{dataset}_{model}_img_cluster.pt')


if __name__ == "__main__":
    args = parse_args()

    model = args.model_name
    dataset = args.dataset
    text = args.text
    n_clusters = args.n_clusters

    if text:
        cluster_text(model, dataset, n_clusters)
    else:
        cluster_img(model, dataset, n_clusters)

    

    
    
    

    
        
    



