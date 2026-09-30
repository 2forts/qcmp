from qcmp.registry import register

from qcmp.comparators.lt_orts_tand import LT_Orts_TemporaryAND
from qcmp.comparators.lt_yuan_qft import LT_Yuan_QFT
from qcmp.comparators.lt_xia_cgc import LT_XIA_CGC
from qcmp.comparators.lt_xia_qip import LT_Xia_QIP_Half
from qcmp.comparators.cmp_oliveira_qbsc import CMP_Oliveira_QBSC
from qcmp.comparators.lt_yuan_adaptive import LT_Yuan_Adaptive
from qcmp.comparators.cmp_kumar_tree_adapted import CMP_Kumar_Tree_Adapted
from qcmp.comparators.cmp_ganapathy_toffoli_adapted import CMP_Ganapathy_Toffoli_Adapted
from qcmp.comparators.lt_orts_constant_tand import LT_Orts_Constant_TAND
from qcmp.comparators.gt_orts_constant_toffoli import GT_Orts_Constant_Toffoli
from qcmp.comparators.gt_donaire_constant_tand import GT_Donaire_Constant_TAND
from qcmp.comparators.lt_berti_qpc import LT_Berti_QPC
from qcmp.comparators.cmp_al_rabadi import CMP_Al_Rabadi

register(LT_Orts_TemporaryAND())
register(LT_Yuan_QFT())
register(LT_XIA_CGC())
register(LT_Xia_QIP_Half())
register(CMP_Oliveira_QBSC())
register(LT_Yuan_Adaptive())
register(CMP_Kumar_Tree_Adapted())
register(CMP_Ganapathy_Toffoli_Adapted())
register(LT_Orts_Constant_TAND())
register(GT_Orts_Constant_Toffoli())
register(GT_Donaire_Constant_TAND())
register(LT_Berti_QPC())
register(CMP_Al_Rabadi())
