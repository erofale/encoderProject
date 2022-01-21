from autoencoder_class import AutoencoderClass
from function_class import Function
from smt.applications import EGO
from smt.sampling_methods import LHS
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error
import random

''' Класс подбора параметров '''
class ParamsSelection():
    def __init__(self):
        pass
    
    def __compare(self, func : Function, orig_data, pred_data):
        '''
        Расчёт отклонения между эталонными и предсказанными данными

        Parameters
        ----------
        func : Function
            Исходная функция.
        orig_data : list
            Эталонный набор данных.
        pred_data : list
            Предсказанный набор данных.
        
        Returns
        -------
        error : float
            Значение ошибки абсолютного отклонения.
        
        '''
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
        dim, irr_dim, _, generator, normalizer = func.get_params()
        error = sys.float_info.max
        rand_samles_count = 200
        rand_data = generator.get_lsh(rand_samles_count, irr_dim)
        norm_data = normalizer.normalize(rand_data)
        hp_list = list()
        for epoch in range(5, 60, h_epoch):
          for batch in  [2**i for i in range(4, 9)]:
            for size in range(dim//2, dim, h_size):
              for percent in np.arange(0.5, 1.0, h_percent):
                  sobol_data = generator.get_sobol(n, irr_dim)
                  random.shuffle(sobol_data)
                  data_train = np.array(sobol_data[0:int(n * percent)])
                  data_test = np.array(sobol_data[int(n * percent):n])
                  model = AutoencoderClass(func, dim, size, enc_type, normalizer)
                  model.fit(data_train, data_test, epoch, batch, True)
                  pred_data = normalizer.renormalize([model.predict(np.array(x).reshape(1,dim))[0] for x in norm_data])
                  cur_error = self.__compare(func, rand_data, pred_data)
                  if cur_error < error:
                    model.save('../../Saved models/Weights/' + f'{func.func_name}_brute_force_{enc_type}_{dim}_{size}.h5')    
                    error = cur_error
                    hp_list.clear()
                    hp_list.append(epoch)
                    hp_list.append(batch)
                    hp_list.append(size)
                    hp_list.append(percent)
        
        with open('../../Saved models/Params/' + f'{func.func_name}_brute_force_{enc_type}_{dim}_{hp_list[2]}.txt', 'w') as f:
            f.write(f'func name: {func.func_name}\nepochs: {hp_list[0]}\nbatch: {hp_list[1]}\nencoded dim: {hp_list[2]}\nsample split: {hp_list[3]}')
        return hp_list, error
    
    
    def ego(self, enc_type : str, func : Function,  n : int, ndoe : int, n_iter : int):
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
        
        dim, irr_dim, _, generator, normalizer = func.get_params()
        rand_samles_count = 200
        rand_data = generator.get_lsh(rand_samles_count, irr_dim)
        norm_data = normalizer.normalize(rand_data)

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
                b_size = int(x[i][1])
                tr_size = (int(n * x[i][3]) // 10) * 10
                
                sobol_data = generator.get_sobol(n, irr_dim)
                random.shuffle(sobol_data)
                data_train = np.array(sobol_data[0 : tr_size])
                data_test = np.array(sobol_data[tr_size : n])             
                
                if (enc_type == 'vae'):
                    while(tr_size % b_size != 0):
                        b_size -= 1
                
                model = AutoencoderClass(func, int(x[i][2]), enc_type)
                model.fit(data_train, data_test, int(x[i][0]), b_size, True)
                model.save('../../Saved models/Weights/' + f'{func.func_name}_ego_{enc_type}_{dim}_{int(x[i][2])}.h5')
                pred_data = normalizer.renormalize([model.predict(np.array(xx).reshape(1,dim))[0] for xx in norm_data])
                res[i] = self.__compare(func, rand_data, pred_data)
            return res
        
        xlimits = np.array([[5, 60], [16, 256], [dim//2, dim-1], [0.5, 1.0]])
        criterion='EI'
        sampling = LHS(xlimits=xlimits, random_state=3)
        xdoe = sampling(ndoe)
        ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)
        x_opt, error, _, _, _ = ego.optimize(fun=predict_params)
        
        x_1, x_2, x_3, x_4 = x_opt
        
        if (enc_type == 'vae'):
            tr_size = (int(n * x_4) // 10) * 10
            while(tr_size % int(x_2) != 0):
                x_2 -= 1
        
        x_opt = [int(x_1), int(x_2), int(x_3), x_4]
        with open('../../Saved models/Params/' + f'{func.func_name}_ego_{enc_type}_{dim}_{x_opt[2]}.txt', 'w') as f:
            f.write(f'func name: {func.func_name}\nepochs: {x_opt[0]}\nbatch: {x_opt[1]}\nencoded dim: {x_opt[2]}\nsample split: {x_opt[3]}')
        return x_opt, error
