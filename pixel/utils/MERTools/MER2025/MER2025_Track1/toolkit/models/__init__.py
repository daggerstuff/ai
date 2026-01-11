"""
get_models: get models and load default configs;
link: https://github.com/thuiar/MMSA-FET/tree/master
"""

import torch.nn as nn

from .attention import Attention
from .attention_topn import Attention_TOPN
from .ef_lstm import EF_LSTM
from .graph_mfn import Graph_MFN
from .lf_dnn import LF_DNN
from .lmf import LMF
from .mctn import MCTN
from .mfm import MFM
from .mfn import MFN
from .misa import MISA
from .mmim import MMIM
from .mult import MULT
from .tfn import TFN


class get_models(nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            "attention": Attention,
            "lf_dnn": LF_DNN,
            "lmf": LMF,
            "misa": MISA,
            "mmim": MMIM,
            "tfn": TFN,
            # 只支持align
            "mfn": MFN,  # slow
            "graph_mfn": Graph_MFN,  # slow
            "ef_lstm": EF_LSTM,
            "mfm": MFM,  # slow
            "mctn": MCTN,  # slow
            # 支持align/unalign
            "mult": MULT,  # slow
            # 支持每个模态选择topn特征输入
            "attention_topn": Attention_TOPN,
        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
