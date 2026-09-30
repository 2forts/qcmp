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
from qcmp.comparators.cmp_thapliyal_tree import CMP_Thapliyal_Tree
from qcmp.comparators.cmp_maity_qcc import CMP_Maity_QCC
from qcmp.comparators.cmp_morrison_rc_tree import CMP_Morrison_RC_Tree
from qcmp.comparators.gt_barui_threshold import GT_Barui_Threshold
from qcmp.comparators.ge_caraiman_threshold_oracle import GE_Caraiman_Threshold_Oracle
from qcmp.comparators.gt_du_nonzero import GT_Du_Nonzero
from qcmp.comparators.cmp_yuan_constant_ancilla import CMP_Yuan_Constant_Ancilla
from qcmp.comparators.cmp_subramanian_tr_half import CMP_Subramanian_TR_Half
from qcmp.comparators.gt_xia_optimized import GT_Xia_Optimized
from qcmp.comparators.cmp_rangaraju_serial import CMP_Rangaraju_Serial

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
register(CMP_Thapliyal_Tree())
register(CMP_Maity_QCC())
register(CMP_Morrison_RC_Tree())
register(GT_Barui_Threshold())
register(GE_Caraiman_Threshold_Oracle())
register(GT_Du_Nonzero())
register(CMP_Yuan_Constant_Ancilla())
register(CMP_Subramanian_TR_Half())
register(GT_Xia_Optimized())
register(CMP_Rangaraju_Serial())
