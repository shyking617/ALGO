#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# 加载项目模块
sys.path.append('/home/dataset-assist-0/taoli/Active_Learning_GO/sphnet-model-data/model_infer/')

from src.utility.hydra_config import Config
from src.training.module import LNNP
from src.training.data import DataModule

def run_inference(model_path=None, test_db_path=None, log_dir=None):
    """
    运行推理的主函数
    
    Args:
        model_path: 模型checkpoint路径
        test_db_path: 测试数据库路径  
        log_dir: 输出目录
    """
    # =====================================================
    # ========== 集中化配置 ================================
    # =====================================================
    USER_CONFIG = {
        # ---------------------- Paths ----------------------
        "log_dir": log_dir or "/home/dataset-assist-0/taoli/Active_Learning_GO/project/model_infer/",
        "ckpt_path": model_path or "/home/dataset-assist-0/taoli/Active_Learning_GO/sphnet-model-data/model-nablaDFT-train100K/test.ckpt",
        # ---------------------- Runtime ---------------------
        "seed": 123,
        "ngpus": 1,
        "precision": 32,
        # ---------------------- Dataset ---------------------
        "data_name": "custom_nabla",
        "basis": "def2-svp",
        "dataset_path": "/home/dataset-assist-0/taoli/Active_Learning_GO/sphnet-model-data/test_2k_conformers.db/data.mdb",
        "val_path": "/home/dataset-assist-0/taoli/Active_Learning_GO/sphnet-model-data/test_2k_conformers.db/data.mdb",
        "test_path": test_db_path or "/home/dataset-assist-0/taoli/Active_Learning_GO/project/stru_test.db/data.mdb",
        "index_path": None,
        "unit": 1,
        # ---------------------- DataLoader -------------------
        "batch_size": 1,
        "dataloader_num_workers": 0,
        # ---------------------- Model Config -----------------
        "model_backbone": "SPHNet",
        "hami_model": {
            "name": "HamiHead_sphnet",
            "irreps_edge_embedding": None,
            "num_layer": 2,
            "max_radius_cutoff": 30,
            "radius_embed_dim": 16,
            "bottle_hidden_size": 32,
        },
        "enable_hami": True,
        "activation": "silu",
        "remove_init": True,
        "remove_atomref_energy": True,
    }

    # 用 Python 字典构造config
    python_cfg = OmegaConf.create(USER_CONFIG)
    print("[INFO] Loaded USER_CONFIG from Python.")
    
    # 合并 Schema（Config类）+ Python 字典
    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, python_cfg)
    OmegaConf.set_struct(cfg, False)
    print("[INFO] Final merged cfg:")
    print(OmegaConf.to_yaml(cfg))

    # 选择 ckpt
    ckpt_path = model_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Using checkpoint: {ckpt_path}")

    # 构建 DataModule & Model
    data = DataModule(cfg)
    model = LNNP.load_from_checkpoint(ckpt_path, config=cfg)
    print(f"[INFO] Loaded model from: {ckpt_path}")

    # 创建 Trainer
    devices = list(range(cfg.ngpus)) if cfg.ngpus > 0 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu" if cfg.ngpus > 0 else "cpu",
        devices=devices,
        default_root_dir=log_dir,
        precision=cfg.precision,
        logger=False,
    )

    if test_db_path:
        # 从db路径提取目录和文件名
        db_dir = os.path.dirname(test_db_path)  # step0.db的目录
        db_parent_dir = os.path.dirname(db_dir)  # step0.db的父目录
        db_folder_name = os.path.basename(db_dir)  # step0.db（文件夹名）

        # 清理扩展名，保留step0
        clean_name = db_folder_name.replace('.db', '')

        # 设置保存目录为step0.db的同级目录
        save_root = os.path.join(db_parent_dir, f"{clean_name}_H_pred")
    else:
        save_root = os.path.join(log_dir, "Predictions-Hami")

    model.test_save_dir = save_root
    os.makedirs(model.test_save_dir, exist_ok=True)
    print(f"[INFO] Hamiltonian prediction will be saved to: {model.test_save_dir}")

    # 推理
    print("[INFO] Running test...")
    results = trainer.test(model, datamodule=data, ckpt_path=ckpt_path)
    print("Test results", results)
    print("[INFO] Test finished.")
    
    return results

# 保留原有的hydra入口（可选）
@hydra.main(version_base="1.3", config_path=None)
def main(cfg: DictConfig):
    """原有的hydra入口，保持向后兼容"""
    model_path = cfg.get("ckpt_path")
    test_path = cfg.get("test_path") 
    log_dir = cfg.get("log_dir")
    run_inference(model_path, test_path, log_dir)

if __name__ == "__main__":
    main()