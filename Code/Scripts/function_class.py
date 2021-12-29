from generator_class import DataGenerator
from normalizer_class import Normalizer
from typing import Tuple, List
import tensorflow as tf

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
            Область определения. Диапазон значений значащих переменных.

        '''
        self.func = func
        self.name = name
        self.dim = dim
        self.irr_dim = irr_dim
        self.data_range = data_range
        self.generator = DataGenerator(self.dim, self.data_range)
        self.normalizer = Normalizer(self.dim, self.irr_dim, self.data_range)
        
    def __call__(self, x):
        '''
        Вызов функции

        Parameters
        ----------
        x : List, ndarray
            Список аргументов функции.

        Returns
        -------
        float
            Значение функции.

        '''
        return self.func(x)
        
    def get_params(self):
        '''
        Получение параметров функции

        Returns
        -------
        int
            Размерность пространства.
        int
            Количество незначащих переменных.
        List[Tuple[float,float]]
            Область определения. Диапазон значений значащих переменных.
        DataGenerator
            Генератор данных для этой функции.
        Normalizer
            Нормировщик данных для этой функции.

        '''
        return self.dim, self.irr_dim, self.data_range, self.generator, self.normalizer
    
    @property
    def func_name(self):
        '''
        Returns
        -------
        str
            Имя функции.

        '''
        return self.name
    

''' Класс тестовых функций '''
class TestFunctions():
    def __init__(self):
        pass
    
    @classmethod
    def get_func_names(self):
        '''
        Получение списка имён тестовых функций

        Returns
        -------
        List[str]
            Список имён.

        '''
        self.functions = {
            'func_1' : self.func_1,
            'func_2' : self.func_2,
            'func_3' : self.func_3,
            'func_4' : self.func_4,
        }
        return list(self.functions.keys())

    @classmethod
    def get_func(self, name : str):
        '''
        Получение функции по её имени

        Parameters
        ----------
        name : str
            Имя функции.

        Returns
        -------
        method
            Вызываемая функция.

        '''
        self.get_func_names()
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
        def f(x):
            return tf.math.pow(x[0] - 1, 2) + tf.math.pow(x[1], 2) + x[2] + 2 * x[3] + tf.math.pow(x[4], 3) + x[5]
        
        data_range = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]
        func = Function(f, 'func_4', 6, 4, data_range)
        return func
