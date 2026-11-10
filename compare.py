import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("timings.csv")))     # each row becomes a dict by column name

print(f"{'label':<24}{'chunker':<11}{'mode':<13}{'chunks':>7}{'embedded':>10}{'total_s':>10}")
for r in rows:
    print(f"{r['label']:<24}{r['chunker']:<11}{r['mode']:<13}{r['total_chunks']:>7}"
          f"{r['embedded']:>10}{float(r['total_s']):>10.3f}")

# Estimate: time ~ overhead + embedded_chunks * cost_per_chunk
base = rows[0]                                       # the first run is the baseline
per_chunk = float(base["embed_store_s"]) / int(base["embedded"])
overhead = float(base["load_s"]) + float(base["chunk_s"])
print(f"\nCost per embedded chunk: {per_chunk:.4f}s | fixed overhead: {overhead:.4f}s")

print("\nPredicted vs measured (incremental runs):")
for r in rows:
    if r["mode"] == "incremental":
        pred = overhead + int(r["embedded"]) * per_chunk
        print(f"  {r['label']:<24} predicted {pred:.3f}s | measured {float(r['total_s']):.3f}s")

# Chart: full rebuilds in red, incremental runs in green
labels = [f"{r['label']}\n({r['chunker']})" for r in rows]
times = [float(r["total_s"]) for r in rows]
colors = ["tab:red" if r["mode"] == "full" else "tab:green" for r in rows]
plt.figure(figsize=(10, 4))
plt.bar(labels, times, color=colors)
plt.ylabel("Ingest time (seconds)")
plt.title("Full rebuild (red) vs incremental (green)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("timing_comparison.png", dpi=150)
plt.show()