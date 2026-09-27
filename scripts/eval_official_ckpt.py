"""Inherited-weight ranking fidelity of an official-code checkpoint (.ckpt or .pth),
using the same 200 architectures and held-out valid batches as official_nb201.py.
Also reports the architecture the search derived (argmax of phi) and its
NAS-Bench-201 test accuracy.

    python scripts/eval_official_ckpt.py --ckpt runs/official/cifar10_s0_gdas_casso_e250.pth
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_nb201 import ranking_fidelity  # noqa: E402
from xautodl.config_utils import dict2config  # noqa: E402
from xautodl.datasets import get_datasets, get_nas_search_loaders  # noqa: E402
from xautodl.models import get_cell_based_tiny_net, get_search_spaces  # noqa: E402
from casso.nb201.api_wrapper import NB201Oracle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
ap.add_argument("--seed", type=int, default=0, help="search seed (selects the 200 architectures)")
ap.add_argument("--data_path", default=os.environ.get("CASSO_DATA", "data/cifar.python"))
ap.add_argument("--autodl_root", default=os.environ.get("AUTODL_ROOT", "AutoDL-Projects"))
ap.add_argument("--oracle", default=os.environ.get("NB201_CACHE", "data/nb201_test_acc_cache.json"))
a = ap.parse_args()
dev = torch.device("cuda")
tr, va, _, ncls = get_datasets(a.dataset, a.data_path, -1)
_, _, vl = get_nas_search_loaders(tr, va, a.dataset,
                                  os.path.join(a.autodl_root, "configs/nas-benchmark/"), 64, 0)
net = get_cell_based_tiny_net(dict2config({"name": "GDAS", "C": 16, "N": 5, "max_nodes": 4,
      "num_classes": ncls, "space": get_search_spaces("cell", "nas-bench-201"),
      "affine": False, "track_running_stats": False}, None)).to(dev)
st = torch.load(a.ckpt, map_location=dev, weights_only=False)
net.load_state_dict(st["net"])
oracle = NB201Oracle(a.oracle)

g = torch.Generator().manual_seed(0)
vidx = vl.sampler.indices
vset = torch.utils.data.Subset(vl.dataset, [vidx[i] for i in torch.randperm(len(vidx), generator=g)[:1280]])
batches = list(torch.utils.data.DataLoader(vset, batch_size=64, shuffle=False))
tau, p, rows = ranking_fidelity(net, oracle, a.dataset, batches, 200, a.seed, dev)
print(f"epoch {st.get('epoch', 'final')}: inherited-weight Kendall tau={tau:.4f} (p={p:.3g}, n={len(rows)})")

arch = net.genotype().tostr()
print(f"derived architecture: {arch}")
print("NAS-Bench-201 test accuracy: " + ", ".join(
    f"{d} {oracle.test_accuracy(arch, d):.2f}" for d in ("cifar10", "cifar100")))
