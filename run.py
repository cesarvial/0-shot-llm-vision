import numpy as np
from sklearn import datasets, cluster
import torch
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
from src.utils import * 
import argparse
from datetime import datetime

DATA_LOCATION = "./data"
AVG_RESULT_LOCATION = "./result"
SEEDWISE_RESULT_LOCATION = "./seedwise"

def move_to_device(obj, device):
    """Recursively move tensors inside obj (tensor, dict, list, tuple) to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--m",
        dest="method",
        help="baseline method",
        default="linear",
        type=str,
    )
    parser.add_argument(
        "--b",
        dest="base_samples",
        help="base_samples",
        default=320,
        type=int,
    )
    parser.add_argument(
        "--q",
        dest="query_samples",
        help="query_samples",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--c",
        dest="clustering_mode",
        help="cluster mode",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--vid_model",
        dest="vid_model",
        help="vid_model",
        default="dinov2",
        type=str,
    )
    parser.add_argument(
        "--text_model",
        dest="text_model",
        help="text_model",
        default="allroberta",
        type=str,
    )
    parser.add_argument(
        "--base_data",
        dest="base_data",
        help="base_data",
        default="coco",
        type=str,
    )
    parser.add_argument(
        "--query_data",
        dest="query_data",
        help="query data",
        default="nocaps",
        type=str,
    )
    parser.add_argument(
        "-str",
        dest="stretch",
        help="stretch",
        action='store_true',
    )
    parser.add_argument(
        "-vtl",
        dest="vtl",
        help="vision to language",
        action='store_true',
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        help="gpu",
        default=7,
        type=int,
    )
    return parser.parse_args()

def extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        # Try common key names first
        for k in ('text_embeddings', 'text_representations', 'text_embeds', 'text_embeds', 
                  'text_embeddings', 'embeddings', 'image_representations', 'image_embeddings',
                  'image_reps', 'emb'):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        # fallback: recursive search
        for v in obj.values():
            t = extract_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = extract_tensor(v)
            if t is not None:
                return t
    return None

if __name__ == "__main__":
    args = parse_args()

    print("here")
    #torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    gpu = 'cuda'
    device = torch.device(gpu)

    target_model = args.vid_model
    source_model = args.text_model
    base_data = args.base_data
    query_data = args.query_data
    vtl = args.vtl
    
    map_loc = 'cpu'  # always safe to load to CPU first

    if not vtl:
        # Text and image "large" objects
        loaded = torch.load(f"{DATA_LOCATION}/{base_data}_{source_model}_text.pt", map_location=map_loc)
        source_base_large = extract_tensor(loaded)
        if source_base_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{base_data}_{source_model}_text.pt")
        source_base_large = source_base_large.to(device)

        loaded = torch.load(f"{DATA_LOCATION}/{base_data}_{target_model}_img.pt", map_location=map_loc)
        target_base_large = extract_tensor(loaded)
        if target_base_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{base_data}_{target_model}_img.pt")
        target_base_large = target_base_large.to(device)

        loaded = torch.load(f"{DATA_LOCATION}/{query_data}_{source_model}_text.pt", map_location=map_loc)
        source_query_large = extract_tensor(loaded)
        if source_query_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{query_data}_{source_model}_text.pt")
        source_query_large = source_query_large.to(device)

        loaded = torch.load(f"{DATA_LOCATION}/{query_data}_{target_model}_img.pt", map_location=map_loc)
        target_query_large = extract_tensor(loaded)
        if target_query_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{query_data}_{target_model}_img.pt")
        target_query_large = target_query_large.to(device)

        # clusters: they may or may not exist; handle missing files gracefully
        src_cluster_path = f"{DATA_LOCATION}/{base_data}_{source_model}_text_cluster.pt"
        tgt_cluster_path = f"{DATA_LOCATION}/{base_data}_{target_model}_img_cluster.pt"
        if os.path.exists(src_cluster_path):
            loaded = torch.load(src_cluster_path, map_location=map_loc, weights_only=False)
            source_base_cluster = extract_tensor(loaded)
            if source_base_cluster is not None:
                source_base_cluster = source_base_cluster.to(device)
            else:
                source_base_cluster = None
        else:
            source_base_cluster = None

        if os.path.exists(tgt_cluster_path):
            loaded = torch.load(tgt_cluster_path, map_location=map_loc, weights_only=False)
            target_base_cluster = extract_tensor(loaded)
            if target_base_cluster is not None:
                target_base_cluster = target_base_cluster.to(device)
            else:
                target_base_cluster = None
        else:
            target_base_cluster = None

    else:
        # vtl branch (vision-to-language) loads reversed roles; same extraction logic
        loaded = torch.load(f"{DATA_LOCATION}/{base_data}_{source_model}_text.pt", map_location=map_loc)
        target_base_large = extract_tensor(loaded)
        if target_base_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{base_data}_{source_model}_text.pt")
        target_base_large = target_base_large.to(device)

        loaded = torch.load(f"{DATA_LOCATION}/{base_data}_{target_model}_img.pt", map_location=map_loc)
        source_base_large = extract_tensor(loaded)
        if source_base_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{base_data}_{target_model}_img.pt")
        source_base_large = source_base_large.to(device)

        loaded = torch.load(f"{DATA_LOCATION}/{query_data}_{source_model}_text.pt", map_location=map_loc)
        target_query_large = extract_tensor(loaded)
        if target_query_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{query_data}_{source_model}_text.pt")
        target_query_large = target_query_large.to(device)

        loaded = torch.load(f"{DATA_LOCATION}/{query_data}_{target_model}_img.pt", map_location=map_loc)
        source_query_large = extract_tensor(loaded)
        if source_query_large is None:
            raise RuntimeError(f"No tensor found in {DATA_LOCATION}/{query_data}_{target_model}_img.pt")
        source_query_large = source_query_large.to(device)

        # cluster paths (vtl)
        tgt_cluster_path = f"{DATA_LOCATION}/{base_data}_{source_model}_text_cluster.pt"
        src_cluster_path = f"{DATA_LOCATION}/{base_data}_{target_model}_img_cluster.pt"
        if os.path.exists(tgt_cluster_path):
            loaded = torch.load(tgt_cluster_path, map_location=map_loc, weights_only=False)
            target_base_cluster = extract_tensor(loaded)
            if target_base_cluster is not None:
                target_base_cluster = target_base_cluster.to(device)
            else:
                target_base_cluster = None
        else:
            target_base_cluster = None

        if os.path.exists(src_cluster_path):
            loaded = torch.load(src_cluster_path, map_location=map_loc, weights_only=False)
            source_base_cluster = extract_tensor(loaded)
            if source_base_cluster is not None:
                source_base_cluster = source_base_cluster.to(device)
            else:
                source_base_cluster = None
        else:
            source_base_cluster = None

    base_samples = args.base_samples
    query_samples = args.query_samples
    clustering_mode = args.clustering_mode
    same = (base_data == query_data)

    method = args.method
    if method == "qap":
        graph_func = lambda a1, a2, a3, a4, a5: None
        retrieval_func = lambda a1: (-1, -1, -1)
        matching_func = qap_matching
    elif method == "linear":
        graph_func = linear_baseline
        retrieval_func = get_retrieval
        matching_func = linear_matching
    elif method == "linear_local_CKA":
        graph_func = linear_local_CKA
        retrieval_func = get_retrieval
        matching_func = linear_matching
    elif method == "kernel_local_CKA":
        graph_func = kernel_local_CKA
        retrieval_func = get_retrieval
        matching_func = linear_matching
    elif method == 'relative':
        graph_func = relative_baseline
        retrieval_func = get_retrieval
        matching_func = linear_matching
    elif method == 'cos':
        graph_func = cos_baseline
        retrieval_func = get_retrieval
        matching_func = linear_matching


    table_seedwise = []
    table = []

    avg_top_1, avg_top_5, avg_top_10, avg_matching = 0, 0, 0, 0
    seeds = [0,1,2]

    print("Starting")
    for seed2 in seeds:
        print("Getting base and queries")
        source_base, target_base, source_query, target_query = get_data_sep(0, seed2, 
                                                                            base_samples, 
                                                                            query_samples, 
                                                                            source_base_large, 
                                                                            target_base_large, 
                                                                            source_query_large, 
                                                                            target_query_large,
                                                                            source_base_cluster,
                                                                            target_base_cluster,
                                                                            clustering_mode, 
                                                                            same,
                                                                            args.stretch)
        print("Got base and queries")

        print("Getting the graph")
        graph = graph_func(source_base, target_base, source_query, target_query, device)
        print("Got the graph")
        
        print("Retrieval")
        top_1, top_5, top_10 = retrieval_func(graph)
        print(top_1, top_5, top_10)
        
        print("Matching")
        matching_acc = matching_func(source_base, target_base, source_query, target_query, graph, device)
        print(matching_acc)

        seedwise_row = [base_data, query_data, source_model, target_model]
        seedwise_row.append(method) # method
        seedwise_row.append(base_samples) # method
        seedwise_row.append(query_samples) # method
        seedwise_row.append(clustering_mode)
        seedwise_row.append(args.stretch)
        seedwise_row.append(0) # seed1
        seedwise_row.append(seed2) # seed2
        seedwise_row.append(matching_acc) # matching
        seedwise_row.append(top_1)
        seedwise_row.append(top_5)
        seedwise_row.append(top_10)

        table_seedwise.append(seedwise_row)

        avg_top_1 += top_1
        avg_top_5 += top_5
        avg_top_10 += top_10
        avg_matching += matching_acc
        
    avg_top_1 = avg_top_1 / len(seeds)
    avg_top_5 = avg_top_5 / len(seeds)
    avg_top_10 = avg_top_10 / len(seeds)
    avg_matching = avg_matching / len(seeds)

    avg_row = [base_data, query_data, source_model, target_model]
    avg_row.append(method) # method
    avg_row.append(base_samples) # method
    avg_row.append(query_samples) # method
    avg_row.append(clustering_mode)
    avg_row.append(args.stretch)
    avg_row.append(avg_matching) # matching
    avg_row.append(avg_top_1)
    avg_row.append(avg_top_5)
    avg_row.append(avg_top_10)

    table.append(avg_row)


    seedwise_col = ["Base data", "Query data", "Source model", "Target model"]
    seedwise_col.append("method") # method
    seedwise_col.append("base_samples") # method
    seedwise_col.append("query_samples") # method
    seedwise_col.append("clustering_mode")
    seedwise_col.append("stretch")
    seedwise_col.append("seed1") # seed1
    seedwise_col.append("seed2") # seed2
    seedwise_col.append("matching_acc") # matching
    seedwise_col.append("top_1")
    seedwise_col.append("top_5")
    seedwise_col.append("top_10")

    avg_col = ["Base data", "Query data", "Source model", "Target model"]
    avg_col.append("method") # method
    avg_col.append("base_samples") # method
    avg_col.append("query_samples") # method
    avg_col.append("clustering_mode")
    avg_col.append("stretch")
    avg_col.append("matching_acc") # matching
    avg_col.append("top_1")
    avg_col.append("top_5")
    avg_col.append("top_10")

    now = datetime.now()
    name = now.strftime("%d-%m-%Y-%H-%M-%S")
    seedwise_df = pd.DataFrame(table_seedwise,
                               columns=seedwise_col)
    seedwise_df.to_csv(f"{SEEDWISE_RESULT_LOCATION}/{name}.csv", index=False)
    
    avg_df = pd.DataFrame(table,
                               columns=avg_col)
    avg_df.to_csv(f"{AVG_RESULT_LOCATION}/{name}.csv", index=False)
    