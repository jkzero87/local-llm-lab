import os, csv, time, tracemalloc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, P, TRUE, POP, GEN, MUT, ELITE, PENALTY = 500, 1000, 20, 20, 15, 0.01, 2, 0.0005


def make_data(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(N, P)).astype(np.float64)
    beta = np.zeros(P)
    idx = rng.choice(P, size=TRUE, replace=False)
    beta[idx] = rng.choice([-1.0, 1.0], size=TRUE) * 2.0
    p = 1.0 / (1.0 + np.exp(-X @ beta))
    y = (rng.random(N) < p).astype(int)
    flip = rng.random(N) < 0.10
    y[flip] = 1 - y[flip]
    return X, y, idx


def method_a(Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    tracemalloc.start()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    acc, f1 = accuracy_score(yte, yhat), f1_score(yte, yhat)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"acc": acc, "f1": f1, "t": time.perf_counter() - t0,
            "mb": peak / 1e6, "snps": P, "found": 0}


def method_b(Xtr, ytr, Xte, yte, true_idx, seed):
    t0 = time.perf_counter()
    tracemalloc.start()
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(POP, P))

    def fitness(ind):
        cols = np.nonzero(ind)[0]
        if len(cols) < 1:
            return 0.0
        Xs = Xtr[:, cols]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        mean = cross_val_score(LogisticRegression(max_iter=1000), Xs, ytr,
                               cv=cv, scoring="accuracy").mean()
        return mean - PENALTY * len(cols)

    for gen in range(GEN):
        fits = np.array([fitness(ind) for ind in pop])
        order = np.argsort(fits)[::-1]
        elite = pop[order[:ELITE]]
        elites = set(map(tuple, elite))
        new = list(elite)
        while len(new) < POP:
            ia, ib = rng.integers(0, POP, 2)
            pa, pb = pop[ia], pop[ib]
            if fitness(pa) < fitness(pb):
                pa, pb = pb, pa
            c = rng.integers(1, P)
            child = np.concatenate([pa[:c], pb[c:]])
            child = child ^ (rng.random(P) < MUT)
            if tuple(child) not in elites:
                elites.add(tuple(child))
                new.append(child)
        pop = np.array(new)
    b = np.argmax(np.array([fitness(ind) for ind in pop]))
    best = pop[b]
    cols = np.nonzero(best)[0]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr[:, cols], ytr)
    yhat = clf.predict(Xte[:, cols])
    acc, f1 = accuracy_score(yte, yhat), f1_score(yte, yhat)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"acc": acc, "f1": f1, "t": time.perf_counter() - t0,
            "mb": peak / 1e6, "snps": len(cols), "found": len(set(cols) & set(true_idx))}


def main():
    rows = []
    for seed in (42, 7, 123):
        X, y, true_idx = make_data(seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
        ra = method_a(Xtr, ytr, Xte, yte)
        ra.update(metodo="A", seed=seed)
        rb = method_b(Xtr, ytr, Xte, yte, true_idx, seed)
        rb.update(metodo="B", seed=seed)
        rows.append(ra); rows.append(rb)

    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "resultados.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "seed", "accuracy", "f1", "tiempo_s",
                    "memoria_mb", "snps_activos", "verdaderos_encontrados"])
        for r in rows:
            w.writerow([r["metodo"], r["seed"], f"{r['acc']:.4f}", f"{r['f1']:.4f}",
                        f"{r['t']:.3f}", f"{r['mb']:.2f}", r["snps"], r["found"]])

    def agg(m):
        s = [r for r in rows if r["metodo"] == m]
        return (np.mean([r["acc"] for r in s]), np.std([r["acc"] for r in s]),
                np.mean([r["f1"] for r in s]), np.std([r["f1"] for r in s]),
                np.mean([r["t"] for r in s]), np.std([r["t"] for r in s]),
                np.mean([r["mb"] for r in s]), np.std([r["mb"] for r in s]))

    a, b = agg("A"), agg("B")
    x = np.arange(3)
    labels = ["Exactitud", "Tiempo (s)", "Memoria (MB)"]
    va = [a[0], a[4], a[6]]; vb = [b[0], b[4], b[6]]
    ea = [a[1], a[5], a[7]]; eb = [b[1], b[5], b[7]]
    wbar = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - wbar / 2, va, wbar, yerr=ea, color="steelblue", label="A: LogisticRegression")
    ax.bar(x + wbar / 2, vb, wbar, yerr=eb, color="seagreen", label="B: Algoritmo genético")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(); ax.set_title("Método A vs B (media ± desv, 3 semillas)")
    fig.tight_layout()
    fig.savefig(os.path.join(base, "grafica.png"), dpi=120)

    print(f"{'Mét':<5}{'Acc':>8}{'F1':>8}{'Tiempo(s)':>10}{'Mem(MB)':>9}{'SNPs':>6}{'Verd':>6}")
    for m, g in (("A", a), ("B", b)):
        snps = np.mean([r["snps"] for r in rows if r["metodo"] == m])
        verd = np.mean([r["found"] for r in rows if r["metodo"] == m]) if m == "B" else 0
        print(f"{m:<5}{g[0]:>8.4f}{g[2]:>8.4f}{g[4]:>10.3f}{g[6]:>9.2f}{snps:>6.0f}{verd:>6.1f}")


if __name__ == "__main__":
    main()
