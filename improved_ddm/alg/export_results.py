import pickle

bsh_locations = [
#    "../model_storage/permuted/baseline/beam_history.pkl",
#    "../model_storage/permuted/long_m2.0/beam_history.pkl",
#    "../model_storage/permuted/long_m1.5/beam_history.pkl",
    "../model_storage/long_permuted/long_m0.0/beam_history.pkl",
    "../model_storage/long_permuted/long_m1.2/beam_s1_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.2/beam_s2_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.2/beam_s3_j0.2_history.pkl",
#    "../model_storage/permuted/long_m1.1/beam_history.pkl",
    "../model_storage/long_permuted/long_m1.01/beam_s1_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.01/beam_s2_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.01/beam_s3_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.01/beam_s4_j0.2_history.pkl",
    "../model_storage/long_permuted_upper_fixed/long_m1.01/beam_s2_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.001/beam_j0.2_history.pkl",
    "../model_storage/long_permuted/long_m1.0001/beam_j0.2_history.pkl",
    # "../model_storage/permuted/d1.0_b0.0/beam_history.pkl",
    # "../model_storage/permuted/d0.1_b0.0/beam_history.pkl",
    # "../model_storage/permuted/d0.01_b0.0/beam_history.pkl",
    # "../model_storage/permuted/d1.0_bn10.0/beam_history.pkl",
    # "../model_storage/permuted/d0.1_bn10.0/beam_history.pkl",
    # "../model_storage/permuted/d0.01_bn10.0/beam_history.pkl",
    # "../model_storage/permuted/m1.2_b0.0/beam_history.pkl",     
    # "../model_storage/permuted/m1.05_b0.0/beam_history.pkl", 
    # "../model_storage/permuted/m1.01_b0.0/beam_history.pkl", 
    # "../model_storage/permuted/m1.005_b0.0/beam_history.pkl", 
    # "../model_storage/permuted/m1.002_b0.0/beam_history.pkl", 
    # "../model_storage/permuted/m1.001_b0.0/beam_history.pkl", 
    # "../model_storage/permuted/m1.0001_b0.0/beam_history.pkl", 
]

types = [
    "baseline",
#    "long_m2.0",
#    "long_m1.5",
    "long_m1.2_s1",
    "long_m1.2_s2",
    "long_m1.2_s3",
#    "long_m1.1",
    "long_m1.01_s1",
    "long_m1.01_s2",
    "long_m1.01_s3",
    "long_m1.01_s4",
    "long_m1.01_s2_uf",
    "long_m1.001",
    "long_m1.0001",
    # "d1.0_b0.0",
    # "d0.1_b0.0",
    # "d0.01_b0.0",
    # "d1.0_bn10.0",
    # "d0.1_bn10.0",
    # "d0.01_bn10.0",
    # "m1.2_b0.0",
    # "m1.05_b0.0",
    # "m1.01_b0.0",
    # "m1.005_b0.0",
    # "m1.002_b0.0",
    # "m1.001_b0.0",
    # "m1.0001_b0.0",
]

results = {}

for loc, typ in zip(bsh_locations, types):
    bsh = pickle.load(open(loc, 'rb'))
    res = {}

    q = [("0", bsh.root)]
    while len(q) > 0:
        pref, node = q.pop()
        if node is None:
            continue

        res[pref] = {
            "test_metrics": node.test_metrics,
            "elbo": node.elbo, 
            "single_log_prob": node.single_log_prob,
            "total_log_prob": node.total_log_prob,
        }

        q.insert(0, (pref+"0", node.child_s0))
        q.insert(0, (pref+"1", node.child_s1))

    results[typ] = res

pickle.dump(results, open("../flattened_results.pkl", "wb"))