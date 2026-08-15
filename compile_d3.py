import numpy as np, glob, os
print(f"{'config':16s} {'diag_med':>8s} {'task0_end':>9s} {'past_end':>8s} {'FINAL_AVG':>9s}")
rows = []
for f in sorted(glob.glob("eigenspectrum/outputs/pgram_d3/*.npz")):
    d = np.load(f)
    key = [k for k in d.files if k.startswith("F_")][0]
    F = d[key]
    n = int(np.sum(~np.isnan(np.diag(F))))
    fa = float(np.nanmean(F[-1, :])) if n == F.shape[0] else float("nan")
    rows.append((fa, os.path.basename(f)[:-4],
                 float(np.nanmedian(np.diag(F))), float(F[-1, 0]),
                 float(np.nanmean(F[-1, :-1])), n))
for fa, tag, dm, t0, pe, n in sorted(rows, key=lambda x: (np.isnan(x[0]), x[0])):
    print(f"{tag:16s} {dm:8.2f} {t0:9.2f} {pe:8.2f} {fa:9.2f}" + ("" if n == 20 else f"  (only {n}/20)"))
print("\nrefs: pgram-perhead r8 11.44 | r16 11.34 | compdual_gramflow 11.44 | adam 28.11")
