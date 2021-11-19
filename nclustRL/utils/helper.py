from collections.abc import Iterable

def loader(cls, module=None):

    """Loads a method from a pointer or a string"""

    if module is None:
        module = []

    if not isinstance(module, Iterable):
        var = module
        module = [var]

    for m in module:
        try:
            return getattr(m, cls) if isinstance(cls, str) else cls

        except AttributeError:
            pass

    raise AttributeError('modules {} have no attribute {}'.format(module, cls))
