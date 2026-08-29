"""训练循环 + CLI —— 唯一允许改优化流程（batching、epoch、早停、多任务编排）的地方。
见 AGENT_RULES.md。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, os, sys, time
import numpy as np

# kit/ 是冻结目录，跟 pipeline/ 是兄弟目录，默认不在 sys.path 上 —— 这两行只是让
# `from data import ...` / `from evaluate import ...` 能找到它，不代表 kit/ 可以改。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kit'))

from data import load
from evaluate import evaluate
from features import encode, FIELDS, IDX
from model import FM

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0, report_test=False):
    """report_test=False（默认）：只算 valid，开发迭代时不碰 test。
    最终汇报一次时才传 report_test=True（对应 train.py --final）。"""
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[IDX['video_id']]] += 1; pos[x[IDX['video_id']]] += x[IDX['label']]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in (('valid', 'test') if report_test else ('valid',)):
        rws = splits[name]
        out[name] = evaluate([x[IDX['user_id']] for x in rws], [x[IDX['label']] for x in rws],
                             [score(x[IDX['video_id']]) for x in rws])
    return out

def run_random(splits, seed=0, report_test=False):
    """report_test=False（默认）：只算 valid，理由同 run_pop。"""
    rng = np.random.default_rng(seed)
    out = {}
    for name in (('valid', 'test') if report_test else ('valid',)):
        rws = splits[name]
        out[name] = evaluate([x[IDX['user_id']] for x in rws], [x[IDX['label']] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           report_test=False):
    """report_test=False（默认）：训练/早停/模型选择只用 valid，函数结束时也不碰 test —
    这是 AGENT_RULES.md 规则 3 的代码落实，不是靠自觉。只有汇报最终数字时才传 True。"""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    out = {'valid': evaluate(uva, yva, m.predict(Xva))}
    if report_test:
        Xte, yte, ute = enc['test']
        out['test'] = evaluate(ute, yte, m.predict(Xte))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--final', action='store_true',
                    help='也算并打印 test（只在最终汇报一次时用；开发迭代时不要加这个 flag）')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': lambda s: run_pop(s, report_test=a.final),
           'random': lambda s: run_random(s, a.seed, report_test=a.final),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                                   report_test=a.final)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in (('valid', 'test') if a.final else ('valid',)):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
