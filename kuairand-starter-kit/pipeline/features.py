"""特征工程 —— 唯一允许改特征的地方。见 AGENT_RULES.md。

只能读 data.py 提供的 splits（每行 schema 见 data.IDX），不允许读 test 的 label
以外的任何东西来构造特征（比如不能用 test 集统计量做分桶边界）。

同一行的输入特征一律走 same_row(x, name)，它会拒绝 data.LEAKY_COLUMNS 里的
曝光后结果列（is_click/is_like/.../play_time_ms/label 等）—— 这些列只能当
同一行的多任务目标，或用于别的行（该用户历史）的序列特征，不能当同一行输入。

对外契约：
    encode(splits) -> (enc, dim)
    enc[name] = (X, y, users)   # X: int32 (N, len(FIELDS)); y: float32 (N,); users: list
    dim                          # 所有 field 的 vocab 总大小，喂给 model.FM(dim, ...)
FIELDS 只是文档用途（当前实现里字段顺序由 raw() 决定），改 raw() 时保持同步。
"""
import numpy as np
from data import IDX, LEAKY_COLUMNS

# 5 个特征域（当前 kit 的 baseline）。想加特征就在 raw() 里加列，并在这里登记名字。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def same_row(x, name):
    """取当前行某一列的值，作为该行的输入特征。曝光后结果列（见 data.LEAKY_COLUMNS）
    在这里直接报错 —— 它们只能当同一行的多任务目标，或用于别的行（历史）的序列特征，
    不能当同一行的输入，否则就是把要预测的答案喂给模型。"""
    if name in LEAKY_COLUMNS:
        raise ValueError(f"{name!r} 是曝光后结果列，不能当同一行的输入特征"
                          f"（同一行只能当多任务目标；历史序列特征要用别的行）")
    return x[IDX[name]]

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    分桶边界只能用 splits['train'] 算，不能用 valid/test。"""
    tr = splits['train']
    edges = _bucket_edges([same_row(x, 'duration_ms') for x in tr])

    def raw(x):
        return [same_row(x, 'user_id'), same_row(x, 'video_id'), same_row(x, 'author_id'),
                same_row(x, 'tab'), str(int(np.searchsorted(edges, same_row(x, 'duration_ms'))))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[IDX['label']]
            users.append(x[IDX['user_id']])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
