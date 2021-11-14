import numpy as np
import random
import pandas as pd
import sobol_seq
import time
from typing import Tuple, List


''' Класс генерации данных '''
class DataGenerator():

    '''
    Parameters
    ----------
    dim: int
        Размерность входных данных.
    val_range : List[Tuple[float,float]]
        Диапазон значений входных данных.
    '''
    def __init__(self, dim : int, val_range : List[Tuple[float,float]]):
        try:
            assert dim  == len(val_range), 'Размерность входных диапазонов не равна входной размерности!'
            self.dim = dim
            self.val_range = val_range
            random.seed(int(time.time()))
        except AssertionError as e:
            raise AssertionError(e.args[0])
    

    def get_random(self, samples_num : int, irrelevant_var_count : int = 0, write_in_file : bool = False) -> List[List[float]]:
        '''
        Рандомная генерация данных.
        Parameters
        ----------
        samples_num : int
            Количество записей.
        irrelevant_var_count : int, optional
            Количество незначащих переменных. По умолчанию 0.
        write_in_file : bool, optional
            Запись в файл. По умолчанию False.

        Returns
        -------
        List[List[float]]
            Список записей.
        '''
        arr = []
        for k in range(samples_num):
            sample = []
            # добавляем существенные переменные
            for i in self.val_range:
                sample.append(random.uniform(i[0], i[1]))
            # добавляем несущественные переменные
            if irrelevant_var_count != 0:
                for i in range(irrelevant_var_count):
                    sample.append(random.uniform(0., 1.))
            arr.append(sample)
        if write_in_file:
            col = [('x' + str(i+1)) for i in range(self.dim + irrelevant_var_count)]
            df = pd.DataFrame(arr, columns=col)
            df.to_csv(f'../../DataSet/random_{self.dim}_{samples_num}_{irrelevant_var_count}.csv')
        return arr
    

    def get_sobol(self, samples_num : int, irrelevant_var_count : int = 0, write_in_file : bool = False) -> List[List[float]]:
        '''
        Генерация данных от 0 до 1 методом Sobol.
        Parameters
        ----------
        samples_num : int
            Количество записей.
        irrelevant_var_count : int, optional
            Количество незначащих переменных. По умолчанию 0.
        write_in_file : bool, optional
            Запись в файл. По умолчанию False.

        Returns
        -------
        List[List[float]]
            Список записей.
        '''
        arr = sobol_seq.i4_sobol_generate(self.dim, samples_num)
        if irrelevant_var_count != 0:
            zeros = [[0] for i in range(irrelevant_var_count)]
            arr = np.insert(arr, obj=self.dim, values=zeros, axis=1)
        if write_in_file:
            col = [('x' + str(i+1)) for i in range(self.dim + irrelevant_var_count)]
            df = pd.DataFrame(arr, columns=col)
            df.to_csv(f'../DataSet/sobol_{self.dim}_{samples_num}_{irrelevant_var_count}.csv')
        return list(arr)
    
    
    def get_from_file(self, filename : str) -> List[List[float]]:
        '''
        Parameters
        ----------
        filename : str
            Имя файла.

        Raises
        ------
        OSError
            Файл не найден.

        Returns
        -------
        List[List[float]]
            Список записей.
        '''
        try:
            return list(pd.read_csv(filename, index_col=0).to_numpy('float32'))
        except OSError as e:
            raise OSError(e.args[0])
