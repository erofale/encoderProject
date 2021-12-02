from smt.applications import EGO
from smt.sampling_methods import LHS
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error
from function_class import Function
import random
from autoencoder_class import AutoencoderClass

''' Класс подбора параметров '''
class ParamsSelection():
    def __init__(self):
        pass
    
    def __compare(self, func : Function, orig_data, pred_data):
         y_orig = [func(x) for x in orig_data]
         y_pred = [func(x) for x in pred_data]
         y_error = mean_absolute_error(y_orig, y_pred)
         return y_error
        
    def brute_force(self, enc_type : str, func : Function, n : int):
        '''
        Полный перебор

        Parameters
        ----------
        enc_type : str
            Тип автоэнкодера.
        func : Function
            Функция для подбора параметров.
        n : int
            Размер генерируемого датасета.

        Returns
        -------
        hp_list : List
            Оптимальный набор параметров.
        error : float
            Оптимальное значение ошибки.

        '''
        
        h_epoch = 5
        h_size = 1
        h_percent = 0.1
        dim, irr_dim, generator, normalizer = func.get_params()
        error = sys.float_info.max
        hp_list = list()
        for epoch in range(5, 60, h_epoch):
          for batch in  [2**i for i in range(4, 9)]:
            for size in range(dim // 2, dim, h_size):
              for percent in np.arange(0.5, 1.0, h_percent):
                  sobol_data = generator.get_sobol(n, irr_dim)
                  random.shuffle(sobol_data)
                  data_train = np.array(sobol_data[0:int(n * h_percent)])
                  data_test = np.array(sobol_data[int(n * h_percent):n])
                  model = AutoencoderClass(func, dim + irr_dim, size, list(['relu', 'sigmoid']), enc_type, normalizer)
                  model.fit(data_train, data_test, epoch, batch, True)
                  rand_data = generator.get_random(100)
                  pred_data = normalizer.renormalize([model.predict(np.array(x).reshape(1,dim + irr_dim))[0] for x in normalizer.normalize(rand_data)])
                  cur_error = self.__compare(func, rand_data, pred_data)
                  if cur_error < error:
                    error = cur_error
                    hp_list.clear()
                    hp_list.append(epoch)
                    hp_list.append(batch)
                    hp_list.append(size)
                    hp_list.append(percent)
        return hp_list, error
    
    
    def ego(self, enc_type : str, func : Function, n : int, ndoe : int, n_iter : int):
        '''
        Метод EGO - эффективная глобальная оптимизация

        Parameters
        ----------
        enc_type : str
            Тип автоэнкодера.
        func : Function
            Функция для подбора параметров.
        n : int
            Размер генерируемого датасета.
        ndoe : int
            Количесто начальных сгенерированных точек.
        n_iter : int
            Максимальное количество итераций алгоритма.

        Returns
        -------
        x_opt : List
            Оптимальный набор параметров.
        error : float
            Оптимальное значение ошибки.

        '''
        
        dim, irr_dim, generator, normalizer = func.get_params()
        
        def predict_params(x):
            ''' 
            x[0] - число эпох
            x[1] - батчсайз
            x[2] - размер сжатия
            x[3] - разбиение выборки
            '''
            count, n_param = x.shape
            res = np.zeros((count,1))
            
            for i in range(count):
              sobol_data = generator.get_sobol(n, irr_dim)
              random.shuffle(sobol_data)
              data_train = np.array(sobol_data[0:int(n * x[i][3])])
              data_test = np.array(sobol_data[int(n * x[i][3]):n])
              model = AutoencoderClass(func, dim + irr_dim, int(x[i][2]), list(['relu', 'sigmoid']), enc_type, normalizer)
              model.fit(data_train, data_test, int(x[i][0]), int(x[i][1]), True)
              rand_data = generator.get_random(100)
              pred_data = normalizer.renormalize([model.predict(np.array(xx).reshape(1,dim + irr_dim))[0] for xx in normalizer.normalize(rand_data)])
              res[i] = self.__compare(func, rand_data, pred_data)
            return res
        
        xlimits = np.array([[5,60], [16,256], [dim//2, dim - 1], [0.5, 1.0]])
        ndoe = 6
        n_iter = 15
        criterion='EI'
        sampling = LHS(xlimits=xlimits, random_state=3)
        xdoe = sampling(ndoe)
        ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)
        x_opt, error, _, _, _ = ego.optimize(fun=predict_params)
        return list(x_opt), error
