from importlib import import_module as _import
def import_module(name):
    prefix="matmul_cost_model._engine."
    return _import(name if name.startswith(prefix) else prefix+name)
