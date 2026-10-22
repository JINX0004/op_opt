"""Static Ascend 910B2 matmul prediction with fixed offline parameters."""
from .api import MatmulCostModel, make_config, expand_config, generate_candidates, supported_tilings, model_info

__version__ = '1.0.0'
__all__ = ['MatmulCostModel', 'make_config', 'expand_config', 'generate_candidates', 'supported_tilings', 'model_info']
