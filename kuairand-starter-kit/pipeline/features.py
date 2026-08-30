"""特征工程 —— 唯一允许改特征的地方。见 AGENT_RULES.md。

kit/data.py 是冻结的 Starter Kit 代码，不能改，这个文件也不从它 import 任何东西 ——
只能读 kit/data.load(splits) 返回的 splits（每行 schema 见下面的 IDX，这是对
kit/data.py 那个定长 tuple 的复述，不是另一份定义，改 kit/data.py 的人如果动了
行 schema，这里也要跟着改）。不允许用 test 的任何东西（包括 label 和统计量）来构造特征
（比如不能用 test 集统计量做分桶边界）。Do not use ANY test-split data to build features.

同一行的输入特征一律走 same_row(x, name)，它会拒绝 LEAKY_COLUMNS 里的曝光后结果列
（is_click/is_like/.../play_time_ms/label 等）—— 这些列只能当同一行的多任务目标，
或用于别的行（该用户历史）的序列特征，不能当同一行输入。kit/data.py 目前只加载了
这 11 列里的 label 本身；其余 10 列如果以后要接进来（多任务/序列特征），只能在这个
文件里自己读 CSV 加进去 —— kit/data.py 不能改 —— 加的时候同样受这条规则约束。

对外契约：
    encode(splits) -> (enc, dim)
    enc[name] = (X, y, users)   # X: int32 (N, len(FIELDS)); y: float32 (N,); users: list
    dim                          # 所有 field 的 vocab 总大小，喂给 model.FM(dim, ...)
FIELDS 不只是文档用途：len(FIELDS) 决定 vocabs 和 X 的第二维（见下面 65/76 行）。
往 raw() 加一列，就必须同时在 FIELDS 登记名字，否则形状不匹配。
FIELDS is NOT documentation: len(FIELDS) sizes vocabs and X. Keep them in sync.
"""
import numpy as np

# kit/data.py 的 load() 返回的定长 tuple 的字段顺序（复述，唯一的 schema 来源仍是
# kit/data.py 本身）。
IDX = {'date': 0, 'user_id': 1, 'video_id': 2, 'author_id': 3,
       'tab': 4, 'duration_ms': 5, 'label': 6}

# 曝光后才产生的结果列（post-impression outcome/feedback）。同一行绝不能拿来预测
# 同一行的 label —— 就算完全只用 train 数据算，同一行内用这些列当输入照样泄漏，
# 因为它们和 label 是同一次曝光的并发结果，不是曝光前已知的信息。目前 kit/data.py
# 只加载了 label；其余列不在当前 IDX 里，same_row() 对它们会直接 KeyError（安全的
# 失败方式）——这个集合列出完整的 11 个名字，是给以后真的把这些列读进来时用的。
LEAKY_COLUMNS = frozenset({
    'label',            # long_view，主任务目标本身
    'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
    'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time',
    'is_profile_enter',
})

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
