from fms5clearer.algo_lib.algo_v2_inpainting.algo_v2_inpainting import Algo as AlgoV2_inp 

def algo_builder(algo_name):
    _ALGO_LIB = {
    'AlgoV2_inp':AlgoV2_inp,
    }
    return _ALGO_LIB[algo_name]
