import numpy as np, glob, os
print(f"{'r':>3s} {'diag_med':>8s} {'task0_end':>9s} {'past_end':>8s} {'FINAL_AVG':>9s}")
rows = {"r2": 2, "r4": 4, "r16": 16, "r32": 32}
out = []
for f in sorted(glob.glob("eigenspectrum/outputs/pgram_sweep/rsweep_*.npz")):
    F = np.load(f)["F_pgram_gramflow"]
    r = rows[os.path.basename(f)[7:-4]]
    out.append((r, float(np.nanmedian(np.diag(F))), float(F[-1, 0]),
                float(np.nanmean(F[-1, :-1])), float(np.nanmean(F[-1, :]))))
F = np.load("eigenspectrum/outputs/pgram_sweep/pg3e-3_aw1e-4_gl1e-4.npz")["F_pgram_gramflow"]
out.append((8, float(np.nanmedian(np.diag(F))), float(F[-1, 0]),
            float(np.nanmean(F[-1, :-1])), float(np.nanmean(F[-1, :]))))
for r, dm, t0, pe, fa in sorted(out):
    print(f"{r:3d} {dm:8.2f} {t0:9.2f} {pe:8.2f} {fa:9.2f}")
