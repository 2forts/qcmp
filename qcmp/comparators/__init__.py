from qcmp.registry import register

from qcmp.comparators.lt_orts_tand import LT_Orts_TemporaryAND
from qcmp.comparators.lt_yuan_qft import LT_Yuan_QFT
from qcmp.comparators.lt_xia_cgc import LT_XIA_CGC
from qcmp.comparators.lt_xia_qip import LT_Xia_QIP_Half
from qcmp.comparators.cmp_oliveira_qbsc import CMP_Oliveira_QBSC
from qcmp.comparators.cmp_al_rabadi import CMP_Al_Rabadi

register(LT_Orts_TemporaryAND())
register(LT_Yuan_QFT())
register(LT_XIA_CGC())
register(LT_Xia_QIP_Half())
register(CMP_Oliveira_QBSC())
register(CMP_Al_Rabadi())
