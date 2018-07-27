# -*- coding: UTF-8 -*-
"""
装饰器:
functools.wraps 保留原函数的元信息
"""

from functools import wraps


def type_assert(cls):
    def decorate(func):
        @wraps(func)
        def wrapp(*args, **kwargs):
            if len(args) >= 1 and not isinstance(args[0], cls):
                raise TypeError("Argument Type Error!")
            return func(*args, **kwargs)
        return wrapp
    return decorate


def type_default(*argv):
    """判断函数的参数默认类型"""
    def decorate(func):
        @wraps(func)
        def wrapp(*args, **kwargs):
            func_args = func.__code__.co_varnames
            args_dict = {argv[i]: argv[i+1] for i in range(0, len(argv), 2)}
            for ky, cls in args_dict.items():
                try:
                    i = func_args.index(ky)
                except:
                    raise AttributeError("argument '%s' is not exist" % ky)
                v = kwargs.get(ky, None) or args[i]
                if not isinstance(v, cls):
                    raise TypeError("argument '%s' type is error" % ky)
            return func(*args, **kwargs)
        return wrapp
    return decorate


@type_default('a', str)
def print_agr(a, b=2):
    print(a, b)


@type_default()
def print_type(a, b):
    print(a, b)


if __name__ == '__main__':
    print_agr('a', 2)
    print_type('a', 2)
