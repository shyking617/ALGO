import os, glob, lmdb, pickle
import numpy as np
from ase.io import read
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
import numpy as np
from torch_scatter import scatter
from sklearn.cluster import KMeans,SpectralClustering,DBSCAN


def num_orbitals_from_Z(Z):
    """按 nabla/def2-SVP 的规则估一个 num_orbitals（和训练时一致）."""
    # 参考QH9的写法: H/He -> 5, 其它 -> 14
    Z = np.asarray(Z, dtype=np.int32)
    return int(sum(5 if z <= 2 else 14 for z in Z))


def xyz_to_data_dict(xyz_file, idx):
    atoms_obj = read(xyz_file)
    atoms = atoms_obj.get_atomic_numbers().astype(np.int32)
    pos = atoms_obj.get_positions().astype(np.float32)  # float32 !!!

    num_nodes = len(atoms)
    norb = num_orbitals_from_Z(atoms)

    # fock 和 fock_init 用全零矩阵（推理不会用）
    Ham = np.zeros((norb, norb), dtype=np.float64)
    Ham_init = np.zeros_like(Ham)

    data_dict = {
        "id": np.int64(idx),
        "num_nodes": np.int32(num_nodes),
        # 必须用 raw bytes
        "atoms": atoms.tobytes(),
        "pos": pos.tobytes(),
        "Ham": Ham.tobytes(),
        "Ham_init": Ham_init.tobytes(),
    }

    return data_dict


def create_lmdb(xyz_path, out_path):  
    # create LMDB environment
    if os.path.isdir(xyz_path):
        files = sorted(glob.glob(f"{xyz_path}/*.xyz"))
    else:
        # 如果传入的是单个 .xyz 文件
        files = [xyz_path]

    env = lmdb.open(out_path, map_size=80 * 1024 * 1024 * 1024)  
    data = []
    for idx, f in enumerate(files):
        d = xyz_to_data_dict(f, idx)
        data.append(d)

    with env.begin(write=True) as txn:  
        txn.put(b"length", pickle.dumps(len(data)))  

        for idx, d in enumerate(data):  
            key = f'{idx}'.encode() # idx.to_bytes(length=4, byteorder='big')  
            value = pickle.dumps(d)  
            txn.put(key, value)  

    env.close()


create_lmdb("/home/dataset-assist-0/taoli/Active_Learning_GO/project/stru_test/h2o.xyz", "/home/dataset-assist-0/taoli/Active_Learning_GO/project/stru_test.db")