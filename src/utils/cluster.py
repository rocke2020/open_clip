from rdkit.ML.Cluster.Butina import ClusterData
from rapidfuzz.distance.Levenshtein import distance as lev_dist
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
import random
from pathlib import Path
import pandas as pd


def cluster_seqs_in_df(df, seq_col_name='Sequence', distThresh=10, extra_filter_func=None,
                       big_cluster_threshold=6, 
                       random_seed=1, 
                       extra_sample_num_in_big_cluster=1):
    clusters = ClusterData(df[seq_col_name].tolist(), len(df), distThresh, distFunc=lev_dist)
    ic(len(df), len(clusters))
    clusters_new = []
    if extra_filter_func is not None:
        for cluster in clusters:
            cluster_new = []
            for idx in cluster:
                if extra_filter_func(df.iloc[idx]):
                    cluster_new.append(idx)
            clusters_new.append(cluster_new)
    else:
        clusters_new = clusters
    
    outs = []
    random.seed(random_seed)
    for cluster in clusters_new:
        if len(cluster) > 0:
            outs.append(cluster[0])
        if len(cluster) > big_cluster_threshold:
            cluster_pos_tmp = list(cluster)
            cluster_pos_tmp.remove(cluster[0])
            idx_plus = random.sample(cluster_pos_tmp, extra_sample_num_in_big_cluster)
            outs.extend(idx_plus)
    ic(len(outs))
    return outs


if __name__ == "__main__":
    afp_root_data_dir = Path('/mnt/sda/bio_drug_corpus/peptides/anti_fungal_peptide')
    train_general_file = afp_root_data_dir / 'train_general.csv'
    df = pd.read_csv(train_general_file)
    cluster_seqs_in_df(df)
    pass