# -*- coding: UTF-8 -*-
"""
python 一切皆对象, 请参考元类 (metaclass) 相关文档
"""

def foo():
    pass


class A(object):
    def foo(self):
        pass


if __name__ == '__main__':
    print(type(foo))
    print(type(foo()))

    # ---------------

    print(type(A.foo))

    print(type(A().foo()))

