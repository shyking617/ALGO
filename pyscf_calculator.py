import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr
import math
import os

class PySCF_Solver_Calculator(Calculator):
    """
    ASE Calculator wrapper for your Solver (DFT/Fock-based)
    """
    implemented_properties = ["energy", "forces"]
    def __init__(
        self,
        solver,
        mode="scf",
        conv_tol=1e-8,
        conv_tol_grad=1e-4,
        
        #---- 传入的 deltaE 区间判据 ----
        deltaE_a=1e-5,
        deltaE_b=1e-3,
        
        #---- 映射参数 ----
        map_A=-1.9,
        map_B=1.55,

        save_path="./",
        step_counter=None,  # 新增：接收步数计数器
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert mode in ["scf", "scf_from_fock", "pred_fock"]

        self.solver = solver
        self.mode = mode
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.step_counter = step_counter  # 保存步数计数器引用

        # ============================================================
        # 自动计算 deltaF_a / deltaF_b
        # ============================================================
        logF_a = (math.log10(deltaE_a) - map_A) / map_B
        logF_b = (math.log10(deltaE_b) - map_A) / map_B

        self.deltaF_a = 10 ** logF_a
        self.deltaF_b = 10 ** logF_b

        print("[INFO] map_b*log10(deltaF) + map_a = log10(deltaE)")
        print(f"[INFO] deltaF_a = {self.deltaF_a:.6e}")
        print(f"[INFO] deltaF_b = {self.deltaF_b:.6e}")

        self.save_path = save_path

    def get_current_step(self):
        """获取当前优化步数"""
        if self.step_counter and 'i' in self.step_counter:
            return self.step_counter['i']
        return 0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # === ASE atoms -> PySCF input ===
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        atom_str = "; ".join(
            f"{sym} {x:.10f} {y:.10f} {z:.10f}"
            for sym, (x, y, z) in zip(symbols, positions)
        )
        self.solver.mol_path = atom_str

        current_step = self.get_current_step()  # 获取当前步数

        # =========================================================
        # (1) 直接 SCF
        # =========================================================
        if self.mode == "scf":
            e_tot, grad, _, _, _, _ = self.solver.run_scf(
                conv_tol=self.conv_tol,
                conv_tol_grad=self.conv_tol_grad,
            )

        # =========================================================
        # (2) 从 F_pred 做初猜 SCF
        # =========================================================
        elif self.mode == "scf_from_fock":
            e_tot, grad, _, F_final, _, _ = self.solver.run_scf_from_fock(
                self.solver.F_pred_path,
                conv_tol=self.conv_tol,
                conv_tol_grad=self.conv_tol_grad,
            )
            # 保存时包含步数信息
            np.save(f"{self.save_path}/F_final_step_{current_step:04d}.npy", F_final)

        # =========================================================
        # (3) pred_fock: ΔF 判定逻辑
        # =========================================================
        elif self.mode == "pred_fock":

            # --- micro SCF: F_pred → dm_new → F_new
            F_new, deltaF, dm_new, mo_e, mo_c, mo_occ = \
                self.solver.micro_scf_from_fock(self.solver.F_pred_path)

            print(f"[INFO] ΔF = {deltaF:.6f}")

            # -------------------------------
            # Case 1: ΔF < a
            # -------------------------------
            if deltaF < self.deltaF_a:
                print("[INFO] ΔF < a: 直接用 F_new 单点能量/力")

                e_tot, grad = self.solver.get_e_f_from_dm(
                    dm_new, mo_e, mo_c, mo_occ
                )
                #np.save(f"{self.save_path}/F_singlepoint_step_{current_step:04d}.npy", F_new)

            # -------------------------------
            # Case 2: a ≤ ΔF < b
            # -------------------------------
            elif deltaF < self.deltaF_b:
                print("[INFO] a <= ΔF < b: 单点能量，然后继续完整 SCF")

                # (1) 先单点得到能量、力
                e_tot, grad = self.solver.get_e_f_from_dm(
                    dm_new, mo_e, mo_c, mo_occ
                )
                # np.save(f"{self.save_path}/F_new_singlepoint_step_{current_step:04d}.npy", F_new)

                # (2) 然后继续完整 SCF
                e2, grad2, _, F_final, _, _ = self.solver.run_scf()
                np.save(f"{self.save_path}/F_final_step_{current_step:04d}.npy", F_final)

            # -------------------------------
            # Case 3: ΔF >= b
            # -------------------------------
            else:
                print("[INFO] ΔF >= b: 从 F_pred 做完整 SCF")

                e_tot, grad, _, F_final, _, _ = self.solver.run_scf_from_fock(
                    self.solver.F_pred_path
                )
                np.save(f"{self.save_path}/F_final_step_{current_step:04d}.npy", F_final)

        # =========================================================
        # 单位转换 + 返回
        # =========================================================
        energy_ev = e_tot * Hartree
        forces_eVA = -grad * Hartree / Bohr

        self.results["energy"] = float(energy_ev)
        self.results["forces"] = np.array(forces_eVA, dtype=float)