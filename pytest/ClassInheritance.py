# -*- coding: UTF-8 -*-
"""
多类继承, 初始化顺序, 普通方法查找顺序(__getattribute__, __get__, __getattr__, 对象, 类)
新式类: 广度优先搜索
旧式类: 深度优先搜索
"""


class A(object):
    def __init__(self, value):
        self.value = value

    def show(self):
        print('A show ', self.value)

    def set(self, value):
        self.value = value
        print('A set ', self.value)


class BA(A):
    def __init__(self, value):
        super(BA, self).__init__(value)
        self.value = value

    def show(self):
        print('BA show ', self.value)


class CA(A):
    def __init__(self, value):
        super(CA, self).__init__(value)
        self.value = value

    def show(self):
        print('CA show ', self.value)

    def set(self, value):
        self.value = value
        print('CA set ', self.value)


class BCA(BA, CA):
    def __init__(self, value):
        super(BCA, self).__init__(value)
        self.value = value


if __name__ == '__main__':
    obj = BCA(1)
    obj.set(2)

