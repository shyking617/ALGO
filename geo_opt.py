import os
import numpy as np
from ase import io
from ase.optimize import BFGS
import stru2mdb as s2m
from model_infer import run_inference
from E_F_solver import Solver
from pyscf_calculator import PySCF_Solver_Calculator

# ------------------------- 配置 ------------------------
model_path = "/home/dataset-assist-0/taoli/Active_Learning_GO/sphnet-model-data/model-nablaDFT-train100K/test.ckpt"
init_xyz = "/home/dataset-assist-0/taoli/Active_Learning_GO/project/stru_test/h2o.xyz"
steps = 10
mode = "scf" # "scf" or "pred_fock" or "scf_from_fock"
work_dir = os.path.join(os.path.dirname(init_xyz), f"{os.path.splitext(os.path.basename(init_xyz))[0]}_opt_{mode}")
frames_dir = os.path.join(work_dir, "frames")
os.makedirs(work_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# ---------------------- create LMDB --------------------
db_path = os.path.join(work_dir, "step0.db")
s2m.create_lmdb(init_xyz, db_path)

# ---------------------- inference ----------------------
print("开始运行模型推理...")
mdb_path = os.path.join(db_path, "data.mdb")
run_inference(model_path, mdb_path, work_dir)
init_fock_path = os.path.join(work_dir, "step0_H_pred", "H_pred.npy")

# ---------------------- 读取原子信息 ----------------------
atoms = io.read(init_xyz)

# ---------------------- 步数计数器 ----------------------
step_counter = {"i": 0}  # 从0开始，对应初始步

# ---------------------- 创建计算器 ----------------------
solver = Solver(init_xyz, init_fock_path, basis="def2-svp", xc="m062x")
calc = PySCF_Solver_Calculator(
    solver, 
    mode=mode, 
    save_path=work_dir,
    step_counter=step_counter  # 传递步数
)
atoms.calc = calc

# ---------------------- 回调函数 ----------------------
def dump_step_data():
    i = step_counter["i"]

    # 1) 保存当前几何到 frames
    xyz_path = os.path.join(frames_dir, f"step_{i:04d}.xyz")
    io.write(xyz_path, atoms)

    # 2) 创建 LMDB 并 run_inference
    db_i = os.path.join(work_dir, f"step{i}.db")
    s2m.create_lmdb(xyz_path, db_i)
    mdb_i = os.path.join(db_i, "data.mdb")

    # 3) 运行模型推理
    H_pred_dir = os.path.join(work_dir, f"step{i}_H_pred")
    os.makedirs(H_pred_dir, exist_ok=True)
    run_inference(model_path, mdb_i, H_pred_dir)

    # 4) 获取预测的 Fock 到 solver 用于计算E 和 F
    solver.F_pred_path = os.path.join(H_pred_dir, "H_pred.npy")

    # 更新计算器中的步数
    step_counter["i"] = i + 1

def dump_step_data_scf():
    """scf 模式专用的回调函数，只保存结构，不进行模型推理"""
    i = step_counter["i"]
    
    # 保存当前几何到 frames
    xyz_path = os.path.join(frames_dir, f"step_{i:04d}.xyz")
    io.write(xyz_path, atoms)
    
    # 更新步数
    step_counter["i"] = i + 1


# ---------------------- 运行优化 ----------------------
dyn = BFGS(atoms, logfile=os.path.join(work_dir, "opt.log"))
if mode == "pred_fock" or mode == "scf_from_fock":
    dyn.attach(dump_step_data, interval=1)
else:
    dyn.attach(dump_step_data_scf, interval=1)

dyn.run(fmax=0.05, steps=steps)

# ---------------------- 输出最终结构 ----------------------
io.write(os.path.join(work_dir, "opt.xyz"), atoms)
print("Optimization finished.")