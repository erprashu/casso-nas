"""One-time extraction of the only NAS-Bench-201 fields the search needs
(arch string by index + test accuracy per dataset) into a small JSON cache.

Every search run previously unpickled the full 1.8GB benchmark (~6GB+ RSS
per process), which made several concurrent runs exhaust system RAM and
trigger the kernel OOM killer. NB201Oracle loads this cache instead when
given a .json path.

Usage:
    python scripts/build_oracle_cache.py \
        --pkl ~/CASSO/data/nasbench201/nasbench201_v1_0-e61699.pkl \
        --out ~/CASSO/data/nasbench201/nb201_test_acc_cache.json
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nas_201_api import NASBench201API  # noqa: E402

DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=os.path.expanduser(
        "~/CASSO/data/nasbench201/nasbench201_v1_0-e61699.pkl"))
    parser.add_argument("--out", default=os.path.expanduser(
        "~/CASSO/data/nasbench201/nb201_test_acc_cache.json"))
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        raw = pickle.load(f)
    api = NASBench201API(raw, verbose=False)

    archs, test_acc = [], {d: [] for d in DATASETS}
    for idx in range(len(api)):
        archs.append(api.arch(idx))
        for d in DATASETS:
            # Same call NB201Oracle.test_accuracy has always used.
            info = api.get_more_info(idx, d, hp="200", is_random=False)
            test_acc[d].append(info.get("test-accuracy"))
        if (idx + 1) % 2000 == 0:
            print(f"{idx + 1}/{len(api)}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"archs": archs, "test_accuracy": test_acc}, f)
    print(f"wrote {len(archs)} archs to {args.out}")


if __name__ == "__main__":
    main()
