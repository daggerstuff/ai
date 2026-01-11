"""
get_models: get models and load default configs;
link: https://github.com/thuiar/MMSA-FET/tree/master
"""

import torch

from .attention import Attention
from .graph_mfn import Graph_MFN
from .lmf import LMF
from .mctn import MCTN
from .mfm import MFM
from .mfn import MFN
from .misa import MISA
from .mmim import MMIM
from .mult import MULT
from .tfn import TFN


class get_models(torch.nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            "attention": Attention,
            "lmf": LMF,
            "misa": MISA,
            "mmim": MMIM,
            "tfn": TFN,
            # 只支持align
            "mfn": MFN,  # slow
            "graph_mfn": Graph_MFN,  # slow
            "mfm": MFM,  # slow
            "mctn": MCTN,  # slow
            # 支持align/unalign
            "mult": MULT,  # slow
        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
