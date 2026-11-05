"""Build the LaTeX tables of report/main.tex from the CSV / JSON files in results/.

    python tools/make_report_tables.py          # writes report/tables/*.tex

Standard library only (no pandas), so it runs anywhere. Every number in the report tables comes from here.
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "tables"

LABEL = {
    "hog_svm": "HOG + linear SVM",
    "resnet_A": "ResNet-18 A ($p=0$)",
    "resnet_p025": "ResNet-18, $p=0.25$",
    "resnet_B": "ResNet-18 B ($p=0.5$)",
    "resnet_p100": "ResNet-18, $p=1.0$",
    "resnet_gray": "ResNet-18, gray background",
    "resnet_blur": "ResNet-18, blurred background",
    "resnet_cutmix": "ResNet-18, CutMix",
    "resnet_rrr1": "ResNet-18, saliency penalty $\\lambda=1$",
    "resnet_rrr10": "ResNet-18, saliency penalty $\\lambda=10$",
    "scratch_A": "ResNet-18 scratch A",
    "scratch_B": "ResNet-18 scratch B",
    "resnet50_A": "ResNet-50 A",
    "resnet50_B": "ResNet-50 B",
    "convnext_A": "ConvNeXt-T A",
    "convnext_B": "ConvNeXt-T B",
    "swin_A": "Swin-T A",
    "swin_B": "Swin-T B",
}
ORDER = ["hog_svm", "resnet_A", "resnet_p025", "resnet_B", "resnet_p100", "resnet_gray", "resnet_blur",
         "resnet_cutmix", "resnet_rrr1", "resnet_rrr10", "scratch_A", "scratch_B", "resnet50_A", "resnet50_B",
         "convnext_A", "convnext_B", "swin_A", "swin_B"]


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}".replace("-", "$-$")


def neg(s):
    return s.replace("-", "$-$")


def pm(mean, std, d=1):
    return neg(f"{100 * mean:.{d}f}") + f" $\\pm$ {100 * std:.{d}f}"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


def mean_std(values):
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def model_of(run):
    return run.rsplit("_seed", 1)[0] if "_seed" in run else run


# ---------------------------------------------------------------- per-run Pet test metrics
per_run = rows(RES / "test" / "summary_per_run.csv")
by_model = defaultdict(list)
for r in per_run:
    run = r[""] if "" in r else r[next(iter(r))]
    by_model[model_of(run)].append({k: float(v) for k, v in r.items() if k not in ("", "model") and v != ""} | {"run": run})


def agg(model, key):
    return mean_std([r[key] for r in by_model[model]])


# ---------------------------------------------------------------- Table: all models on Pet test
lines = []
for m in ORDER:
    if m not in by_model:
        continue
    cells = [LABEL[m], str(len(by_model[m]))]
    for key in ("acc_original", "acc_swapped", "acc_fgonly", "acc_bgonly"):
        cells.append(pm(*agg(m, key)))
    for key in ("gap_swap", "gap_fg", "flip_swap"):
        cells.append(pm(*agg(m, key)))
    lines.append(" & ".join(cells) + " \\\\")
    if m in ("hog_svm", "resnet_p100", "resnet_rrr10", "scratch_B"):
        lines.append("\\midrule")
if lines[-1] == "\\midrule":
    lines.pop()
write("main_all", "\n".join(lines) + "\n")

# F1 table (macro F1 per condition)
lines = []
for m in ORDER:
    if m not in by_model:
        continue
    cells = [LABEL[m]] + [pm(*agg(m, f"f1_{c}")) for c in ("original", "swapped", "fgonly", "bgonly")]
    lines.append(" & ".join(cells) + " \\\\")
write("f1_all", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: architectures with bootstrap CI
lines = []
for r in rows(RES / "analysis" / "architectures.csv"):
    cells = [r["family"] if r["variant"] == "A" else "", r["variant"],
             pm(float(r["acc_original"]), float(r["acc_original_std"])),
             pm(float(r["acc_swapped"]), float(r["acc_swapped_std"])),
             pm(float(r["gap_swap"]), float(r["gap_swap_std"])),
             pm(float(r["flip_swap"]), float(r["flip_swap_std"])),
             pct(r["acc_bgonly"])]
    if r["variant"] == "B":
        cells.append(f"{pct(r['gapA_minus_gapB'])} [{pct(r['ci_low'])}, {pct(r['ci_high'])}]")
    else:
        cells.append("")
    lines.append(" & ".join(cells) + " \\\\")
    if r["variant"] == "B":
        lines.append("\\addlinespace")
lines.pop()
write("architectures", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: methods
lines = []
for r in rows(RES / "analysis" / "methods.csv"):
    name = (r["method"].replace("λ", "$\\lambda$").replace("Places365 swap (B)", "\\textbf{Places365 swap (B, ours)}"))
    cells = [name,
             pm(float(r["acc_original"]), float(r["acc_original_std"])),
             pm(float(r["acc_swapped"]), float(r["acc_swapped_std"])),
             pm(float(r["acc_fgonly"]), float(r["acc_fgonly_std"])),
             pm(float(r["gap_swap"]), float(r["gap_swap_std"])),
             pm(float(r["flip_swap"]), float(r["flip_swap_std"]))]
    lines.append(" & ".join(cells) + " \\\\")
write("methods", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: ablation over p
with open(RES / "analysis" / "ablation_p.csv", newline="") as fh:
    ab = list(csv.reader(fh))[3:]   # three header rows (pandas multi-index)
lines = []
for vals in ab:
    p, ao, aos, asw, asws, g, gs = vals[:7]
    model = {"0.0": "resnet_A", "0.25": "resnet_p025", "0.5": "resnet_B", "1.0": "resnet_p100"}[p]
    flip = agg(model, "flip_swap")
    fg = agg(model, "acc_fgonly")
    lines.append(" & ".join([p, pm(float(ao), float(aos)), pm(float(asw), float(asws)), pm(*fg),
                             pm(float(g), float(gs)), pm(*flip)]) + " \\\\")
write("ablation", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: Stanford Dogs
lines = []
prev = None
for r in rows(RES / "analysis" / "stanford_dogs.csv"):
    group = r["group"].replace("λ", "$\\lambda$")
    if group in ("CutMix", "Blurred background", "Gray background") or "Saliency" in r["group"]:
        if r["variant"] == "A":
            continue          # the A row of every method is ResNet-18 A, already listed
    if prev is not None and group != prev:
        lines.append("\\addlinespace")
    prev = group
    label = LABEL.get(r["model"], r["model"])
    cells = [label,
             pm(float(r["acc_full"]), float(r["acc_full_std"])),
             pm(float(r["acc_crop"]), float(r["acc_crop_std"])),
             pm(float(r["acc_dog_full"]), float(r["acc_dog_full_std"])),
             pct(r["oxford_acc_original"])]
    if r["B_minus_A_full"]:
        cells.append(neg(f"{float(r['B_minus_A_full']) * 100:+.1f} [{float(r['ci_low']) * 100:+.1f}, {float(r['ci_high']) * 100:+.1f}]"))
    else:
        cells.append("")
    lines.append(" & ".join(cells) + " \\\\")
write("stanford_dogs", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: Waterbirds
wb = rows(RES / "waterbirds" / "test_per_run.csv")
groups = ["avg_acc", "worst_group_acc", "landbird on land", "landbird on water", "waterbird on land", "waterbird on water"]
lines = []
for m, name in (("waterbirds_A", "A: plain fine-tuning"), ("waterbirds_B", "B: random background swap")):
    rs = [r for r in wb if r["model"] == m]
    cells = [name] + [pm(*mean_std([float(r[g]) for r in rs])) for g in groups]
    lines.append(" & ".join(cells) + " \\\\")
write("waterbirds", "\n".join(lines) + "\n")
lines = []
for r in wb:
    lines.append(" & ".join([r["model"].replace("waterbirds_", ""), r["seed"]] + [pct(r[g]) for g in groups]) + " \\\\")
write("waterbirds_per_seed", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: probing
pr = {r["model"]: r for r in rows(RES / "analysis" / "probing.csv")}
REF = [("ResNet-18", "imagenet_resnet18", "resnet_A", "resnet_B"),
       ("ResNet-18 scratch", "random_resnet18", "scratch_A", "scratch_B"),
       ("ResNet-50", "imagenet_resnet50", "resnet50_A", "resnet50_B"),
       ("ConvNeXt-T", "imagenet_convnext_tiny", "convnext_A", "convnext_B"),
       ("Swin-T", "imagenet_swin_t", "swin_A", "swin_B")]
lines = []
for fam, ref, a, b in REF:
    lines.append(" & ".join([fam + (" (random init.)" if "scratch" in fam else " (ImageNet)"), pct(pr[ref]["mean"]),
                             pm(float(pr[a]["mean"]), float(pr[a]["std"])), pm(float(pr[b]["mean"]), float(pr[b]["std"]))]) + " \\\\")
write("probing_arch", "\n".join(lines) + "\n")
lines = []
for m in ("resnet_A", "resnet_p025", "resnet_B", "resnet_p100", "resnet_gray", "resnet_blur", "resnet_cutmix",
          "resnet_rrr1", "resnet_rrr10"):
    lines.append(f"{LABEL[m]} & {pm(float(pr[m]['mean']), float(pr[m]['std']))} \\\\")
write("probing_methods", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: training summary (validation)
hist = defaultdict(list)
for f in sorted((RES / "runs").glob("*_seed*/history.json")):
    h = json.loads(f.read_text())
    best = max(h, key=lambda r: r["val_acc_original"])
    hist[model_of(f.parent.name)].append({
        "val_o": best["val_acc_original"], "val_s": best["val_acc_swapped"], "ep": best["epoch"] + 1,
        "n": len(h), "min": sum(r["time_s"] for r in h) / 60, "tr": h[-1]["train_acc"]})
lines = []
for m in ORDER:
    if m not in hist:
        continue
    hs = hist[m]
    lines.append(" & ".join([LABEL[m], str(hs[0]["n"]), ", ".join(str(x["ep"]) for x in hs),
                             pm(*mean_std([x["val_o"] for x in hs])), pm(*mean_std([x["val_s"] for x in hs])),
                             pct(st.mean(x["tr"] for x in hs)), f"{st.mean(x['min'] for x in hs):.1f}"]) + " \\\\")
write("training", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table: per-breed gaps (largest |gap_A|)
pb = rows(RES / "analysis" / "per_breed_gap.csv")
pb.sort(key=lambda r: -abs(float(r["gap_A"])))
lines = []
for r in pb[:12]:
    lines.append(" & ".join([r["breed"].replace("_", " "), r["species"],
                             neg(f"{float(r['gap_A']) * 100:+.1f}"), neg(f"{float(r['gap_B']) * 100:+.1f}")]) + " \\\\")
write("per_breed", "\n".join(lines) + "\n")
sp = {}
for s in ("cat", "dog"):
    xs = [r for r in pb if r["species"] == s]
    sp[s] = (st.mean(float(r["gap_A"]) for r in xs), st.mean(float(r["gap_B"]) for r in xs), len(xs))

# ---------------------------------------------------------------- Table: Grad-CAM share on the pet
gc = defaultdict(list)
area = None
for r in rows(RES / "analysis" / "gradcam_foreground_share.csv"):
    gc[(r["model"], r["condition"])].append(float(r["cam_on_pet"]))
    area = float(r["pet_area"])
lines = []
for m in ("resnet_A", "resnet_B", "resnet_gray", "resnet_blur", "resnet_cutmix", "resnet_rrr1", "resnet_rrr10"):
    if (m, "original") in gc:
        lines.append(" & ".join([LABEL[m], f"{st.mean(gc[(m, 'original')]):.3f}", f"{st.mean(gc[(m, 'swapped')]):.3f}"]) + " \\\\")
write("gradcam", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- Appendix: every Pet run
lines = []
for m in ORDER:
    for r in sorted(by_model.get(m, []), key=lambda r: r["run"]):
        seed = r["run"].rsplit("_seed", 1)[1] if "_seed" in r["run"] else "--"
        lines.append(" & ".join([LABEL[m], seed] + [pct(r[k]) for k in ("acc_original", "acc_swapped", "acc_fgonly",
                                                                      "acc_bgonly", "gap_swap", "flip_swap")]) + " \\\\")
write("per_run", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- numbers quoted in the text
boot = json.loads((RES / "analysis" / "bootstrap_gap_difference.json").read_text())
macros = {
    "bootDiff": pct(boot["difference"]), "bootLo": pct(boot["ci_low"]), "bootHi": pct(boot["ci_high"]),
    "gradcamArea": f"{area:.2f}",
    "catGapA": pct(sp["cat"][0]), "catGapB": pct(sp["cat"][1]), "dogGapA": pct(sp["dog"][0]), "dogGapB": pct(sp["dog"][1]),
}
write("numbers", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
