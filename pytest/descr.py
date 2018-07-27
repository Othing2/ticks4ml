# -*- coding: UTF-8 -*-
"""
描述器: 定义类属性查找顺序, 描述器属性被优先查找, 实现__get__(), __set__(), __del__() 任一方法的都可称为描述器
可参考 @property 源码
描述器可分为'资料描述器'和'非资料描述器'
"""


class classproperty(object):
    def __init__(self, fget=None, fset=None, fdel=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel

    def __get__(self, instance, owner):
        if instance is None:
            return owner

        return self.fget(owner)


class MyClass(object):

    @classproperty
    def name(cls):
        return cls.__name__

    @property
    def x(self):
        return self._data

    @x.setter
    def x(self, value):
        self._data = value

    @x.deleter
    def x(self):
        del self._data

    def __init__(self, val):
        self._data = val
        self.x = 3
        self.name = 'pytest'

    def __getattr__(self, item):
        if not self.__dict__.get(item, None):
            self.__dict__[item] = classproperty()
            print("New ClassProperty: %s" % item)
        return self.__dict__[item]


if __name__ == '__main__':
    import os, sys
    s = MyClass(99)
    print(s.x)
    print(s.name)
    print(s.sss)
    print(s.__dict__)

