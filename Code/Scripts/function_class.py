from generator_class import DataGenerator
from normalizer_class import Normalizer
from typing import Tuple, List
import tensorflow as tf
import math

''' Класс функции '''
class Function():
    def __init__(self, func, name : str, dim : int, irr_dim : int, data_range : List[Tuple[float,float]]):
        '''
        Объект функции включает в себя:

        Parameters
        ----------
        func : function, lambda
            Непостредственно функция.
        name : str
            Название функции.
        dim : int
            Размерность пространства.
        irr_dim : int
            Количество незначащих переменных. Соответственно, размерность функции (dim - irr_dim).
        data_range : List[Tuple[float,float]]
            Область опредеоления. Диапазон значений значащих переменных.

        '''
        self.func = func
        self.name = name
        self.dim = dim
        self.irr_dim = irr_dim
        self.data_range = data_range
        self.generator = DataGenerator(self.dim, self.data_range)
        self.normalizer = Normalizer(self.dim, self.irr_dim, self.data_range)
        
    def __call__(self, x):
        return self.func(x)
        
    def get_params(self):
        return self.dim, self.irr_dim, self.data_range, self.generator, self.normalizer
    
    @property
    def func_name(self):
      return self.name
    

''' Класс тестовых функций '''
class TestFunctions():
    def __init__(self):
        pass
    
    @classmethod
    def get_func(self, name : str):
        self.functions = {
            'func_1' : self.func_1,
            'func_2' : self.func_2,
            'func_3' : self.func_3,
            'func_4' : self.func_4,
            'func_5' : self.func_5,
            'func_6' : self.func_6
        }
        return self.functions[name](self)

    def func_1(self):
        def f(x):
            return tf.math.pow(x[0],2) + tf.math.pow(x[1],2) + tf.math.pow(x[2],2) + tf.math.pow(x[3],2) + tf.math.pow(x[4],2) + tf.math.pow(x[5],2) + tf.pow(x[6],2) + tf.pow(x[7],2)
        
        data_range = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100),  (0, 100), (0, 100)]
        func = Function(f, 'func_1', 8, 0, data_range)
        return func
        
    def func_2(self):
        def f(x):
            return tf.math.pow(x[0],4) + 4 * tf.math.pow(x[0],3) * x[1] + 6 * tf.math.pow(x[0],2) + tf.math.pow(x[1],2) + 4 * x[0] * tf.math.pow(x[1],3) + tf.math.pow(x[1],4)
        
        data_range = [(0, 25), (0, 25)]
        func = Function(f, 'func_2', 2, 2, data_range)
        return func
    
    def func_3(self):
        def f(x):
            return tf.math.pow(x[0] - 100, 2) + tf.math.pow(x[1] + 3, 2) + 5 * tf.math.pow(x[2] + 10, 2)
        
        data_range = [(0, 100), (0, 100), (0, 100)]
        func = Function(f, 'func_3', 3, 3, data_range)
        return func
    
    def func_4(self):
        def f(t):
            def x1(t):
                return t
            def x2(t):
                return tf.math.pow(tf.math.sin(t), 2)
            def x3(t):
                return tf.math.pow(tf.math.cos(t), 2)
            def x4(t):
                return tf.math.pow(t, 2)
            def x5(t):
                return 2 * t - 1
            return x1(t[0]) + x2(t[0]) + x3(t[0]) + x4(t[0]) + x5(t[0])
        
        data_range = [(0, math.pi / 2)]
        func = Function(f, 'func_4', 1, 0, data_range)
        return func
    
    def func_5(self):
        def f(t):
            def x1(t):
                return t
            def x2(t):
                return tf.math.pow(tf.math.sin(t), 2)
            def x3(t):
                return tf.math.pow(tf.math.cos(t), 2)
            def x4(t):
                return 3 * t - 5
            def x5(t):
                return t - 2
            def x6(t):
                return tf.math.pow(tf.math.sin(t), 4)
            def x7(t):
                return tf.math.pow(tf.math.cos(t), 4)
            return tf.math.pow(x1(t[0]), 2) + x2(t[0]) + 3 * x3(t[0]) + x4(t[0]) + tf.math.pow(x5(t[1]), 2) + x6(t[1]) + x7(t[1])
        
        data_range = [(0, math.pi / 2), (0, math.pi / 2)]
        func = Function(f, 'func_5', 2, 0, data_range)
        return func
    
    def func_6(self):
        def f(x):
            return tf.math.pow(x[0] - 1, 2) + tf.math.pow(x[1], 2) + x[2] + 2 * x[3] + tf.math.pow(x[4], 3) + x[5]
        
        data_range = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]
        func = Function(f, 'func_6', 6, 4, data_range)
        return func
