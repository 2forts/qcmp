from qcmp.registry import register

from qcmp.comparators.lt_orts_tand import LT_Orts_TemporaryAND
from qcmp.comparators.lt_yuan_qft import LT_Yuan_QFT

register(LT_Orts_TemporaryAND())
register(LT_Yuan_QFT())