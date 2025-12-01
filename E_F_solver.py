import numpy as np
from numpy.linalg import eigh
from pyscf import gto, dft
from pyscf.scf import hf
import time


class Solver:
    """
    Solver: A utility class for handling Fock-based DFT workflows with PySCF.
    Supports:
      - building molecule & DFT object
      - density matrix reconstruction from Fock matrix
      - energy/gradient evaluation
      - orbital gradient evaluation
      - full SCF run and saving of converged matrices
    """

    def __init__(
        self,
        mol_path: str,
        F_pred_path: str,
        basis: str = "def2-svp",
        charge: int = 0,
        spin: int = 0,
        xc: str = "m062x",
    ):
        """Initialize calculation parameters."""
        self.mol_path = mol_path
        self.F_pred_path = F_pred_path
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.xc = xc

        # Loaded / constructed objects
        self.F_pred = np.load(F_pred_path)
        self.mol = None
        self.mf = None

    def build_mf(self, verbose: int = 5):
        """Build molecule and mean-field (RKS) object."""
        mol = gto.Mole()
        mol.build(
            atom=self.mol_path,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=verbose,
        )
        self.mol = mol

        mf = dft.RKS(mol)
        mf.xc = self.xc
        mf.verbose = verbose
        self.mf = mf
        return mf

    def dm_from_Fock(self, F_path: str):
        """
        从给定的 Fock 矩阵与重叠矩阵构造密度矩阵。

        Returns:
            dm: 密度矩阵
            mo_coeff: 轨道系数矩阵
            mo_energy: 轨道能量
            mo_occ: 轨道占据
        """
        if self.mf is None:
            self.build_mf()
        mf = self.mf
        S = self.mf.get_ovlp()
        F = np.load(F_path)

        print(f"[INFO] Loaded Fock shape: {F.shape}")
        mo_energy, mo_coeff = hf.eig(F, S)
        mo_occ = self.mf.get_occ(mo_energy)

        dm = self.mf.make_rdm1(mo_coeff, mo_occ)
        return dm, mo_coeff, mo_energy, mo_occ

    def get_e_f_from_dm(
        self,
        dm: np.ndarray,
        mo_energy: np.ndarray,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
    ):
        """Compute energy and gradient from given density matrix."""
        if self.mf is None:
            raise RuntimeError("MF object not built. Call build_mf() first.")

        self.mf.mo_coeff = mo_coeff
        self.mf.mo_energy = mo_energy
        self.mf.mo_occ = mo_occ

        e_tot = self.mf.energy_tot(dm=dm)
        grad_obj = self.mf.nuc_grad_method()
        grad = grad_obj.kernel()
        return e_tot, grad

    def get_orb_grad(
        self,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        F: np.ndarray,
    ):
        """Compute orbital gradient (virtual-occupied coupling)."""
        occ_idx = mo_occ > 0
        vir_idx = ~occ_idx

        orb_grad = (
            mo_coeff[:, vir_idx].conj().T @ (F @ mo_coeff[:, occ_idx]) * 2.0
        )
        return orb_grad, orb_grad.ravel()

    def run_scf(self, conv_tol: float = 1e-8, conv_tol_grad: float = 1e-4):
        """Run a standard SCF cycle and save converged Fock matrix."""
        mf = self.build_mf()
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = conv_tol_grad

        start = time.time()
        mf.kernel()
        end = time.time()

        print(f"[INFO] SCF finished in {end - start:.2f} s")

        # Extract useful quantities
        e_tot = mf.e_tot
        grad = mf.nuc_grad_method().kernel()
        H_core = mf.get_hcore()
        Fock = mf.get_fock()
        C = mf.mo_coeff
        eps = mf.mo_energy

        # np.save("conv_F.npy", Fock)
        # print("[INFO] Saved converged Fock matrix -> conv_F.npy")

        return e_tot, grad, H_core, Fock, C, eps

    def run_scf_from_fock(
        self,
        F_path: str,
        conv_tol: float = 1e-8,
        conv_tol_grad: float = 1e-4,
    ):
        """
        使用给定 Fock 矩阵产生的初始密度矩阵 dm0 开始 SCF 迭代。
        参数:
            F_init_path: 初始 Fock 矩阵 (.npy)
        """
        mf = self.build_mf()
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = conv_tol_grad

        F_init = np.load(F_path)
        S = mf.get_ovlp()

        print(f"[INFO] Loaded initial Fock shape: {F_init.shape}")

        from pyscf.scf.hf import eig
        mo_energy0, mo_coeff0 = eig(F_init, S)
        mo_occ0 = mf.get_occ(mo_energy0)
        dm = mf.make_rdm1(mo_coeff0, mo_occ0)
        print("[INFO] Using dm0 from provided Fock as SCF initial guess.")

        # (6) run SCF from dm0 (PySCF official)
        start = time.time()
        e_tot = mf.kernel(dm)
        end = time.time()

        if not mf.converged:
            print("[WARN] SCF did not converge.")

        print(f"[INFO] SCF finished in {end - start:.2f} s")

        # Extract useful outputs
        grad = mf.nuc_grad_method().kernel()
        H_core = mf.get_hcore()
        Fock = mf.get_fock()
        C = mf.mo_coeff
        eps = mf.mo_energy

        # Save the converged Fock
        # np.save("conv_F.npy", Fock)
        # print("[INFO] Saved converged Fock matrix -> conv_F.npy")

        return e_tot, grad, H_core, Fock, C, eps

    def micro_scf_from_fock(
        self,
        F_path: str,
    ):
        """
        给定一个 Fock 矩阵 F_in，执行一次 SCF micro-iteration：
        1) 对角化 F_in 得到 mo_coeff, mo_energy
        2) 生成新的密度矩阵 dm_new
        3) 计算新的 vhf
        4) 得到下一步的 Fock: F_new = hcore + vhf
        5) 计算 ΔH = ||F_new - F_in||_F
        Returns:
            F_new: 下一步 Fock
            deltaH: Frobenius norm (F_new - F_in)
            dm_new
            mo_energy, mo_coeff, mo_occ
        """

        if self.mf is None:
            self.build_mf()
        mf = self.mf
        mol = self.mol
        S = mf.get_ovlp()
        h1e = mf.get_hcore()

        # diagonalize Fock matrix
        F_in = np.load(F_path)
        mo_energy, mo_coeff = hf.eig(F_in, S)
        mo_occ = mf.get_occ(mo_energy)

        # build new density ---
        dm_new = mf.make_rdm1(mo_coeff, mo_occ)

        # compute new V_H + V_xc ---
        #       (V_xc automatically handled for DFT)
        vhf_new = mf.get_veff(mol, dm_new)

        # new Fock ---
        F_new = mf.get_fock(h1e, S, vhf_new, dm_new)

        # delta H ---
        deltaH = np.linalg.norm(F_new - F_in)

        return F_new, deltaH, dm_new, mo_energy, mo_coeff, mo_occ


if __name__ == "__main__":
    
    solver = Solver(
        mol_path="h2o.xyz",
        F_pred_path="Fock.npy",
        basis="def2-svp",
        xc="m062x"
    )
    solver.build_mf()
    # ============================================================
    # (1) 运行完整 SCF（从头开始，正常 PySCF）
    # ============================================================
    # e_tot, grad, H, F, C, eps = solver.run_scf()

    # ============================================================
    # (2) 使用给定 Fock 作为初始 guess 运行 SCF（由外部模型提供）
    # ============================================================
    # e_tot2, grad2, H2, F2, C2, eps2 = solver.run_scf_from_fock("Fock.npy")

    # ============================================================
    # (3) 仅根据 Fock.npy 恢复密度矩阵、MO 能量/系数/占据
    # ============================================================
    # dm, mo_coeff, mo_energy, mo_occ = solver.dm_from_Fock("Fock.npy")

    # ============================================================
    # (4) 给定密度矩阵直接计算 E_tot 与 梯度（单点计算）
    # ============================================================
    # e_dm, grad_dm = solver.get_e_f_from_dm(dm, mo_energy, mo_coeff, mo_occ)

    # ============================================================
    # (5) 计算轨道梯度（MO 的虚-占耦合矩阵）
    # ============================================================
    # orb_grad, orb_grad_flat = solver.get_orb_grad(mo_coeff, mo_occ, solver.F_pred)

    # ============================================================
    # (6) 执行一次 micro-SCF 步（单步：F_in → dm → vhf → F_out）
    # ============================================================
    # F_new, deltaH, dm_new, mo_e_new, mo_c_new, mo_occ_new = solver.micro_scf_from_fock("Fock.npy")