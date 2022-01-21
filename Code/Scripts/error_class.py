from autoencoder_class import AutoencoderClass
from function_class import Function
from generator_class import DataGenerator
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

''' Класс подсчёта ошибки '''
class ErrorCalculate():
    def __init__(self, func : Function):
        self.func = func
        self.dim, self.irr_dim, self.data_range, self.generator, self.normalizer = self.func.get_params()

    def calculate(self, aec : AutoencoderClass, num_samples : int = 1000):
        '''
          Расчёт ошибки: средней и по сегментам
    
          Parameters
          ----------
          aec : AutoencoderClass
              Объект обученного автоэнкодера.
          num_samples : int
              Размер выборки для подсчёта средней ошибки.
    
          Returns
          -------
          y_mean_error : float
              Средняя ошибка по всем предсказаниям.
          fig : pyplot
              График ошибок по сегментам.
        '''
        rand_data = self.generator.get_lsh(num_samples, self.irr_dim)
        norm_data = self.normalizer.normalize(rand_data)
        pred_data = self.normalizer.renormalize([aec.predict(np.array(xx).reshape(1,self.dim))[0] for xx in norm_data])
        y_orig = [self.func(x) for x in rand_data]
        y_pred = [self.func(x) for x in pred_data]
        # средняя ошибка по всем предсказаниям
        y_mean_error = mean_absolute_error(y_orig, y_pred)
    
        # средняя ошибка по кластерам
        fig = plt.figure(figsize = (4 * self.dim, 4))
        samp_n = num_samples // 10
        for i in range(self.dim):
            a, b = self.data_range[i]
            h = (b - a) / 10
            err = []
            names = []
            for k in np.arange(a, b, h):
                d_r = self.data_range.copy()
                d_r[i] = (k, k + h)
                names.append(f'{k:.1f}-{k+h:.1f}')
                cur_generator = DataGenerator(self.dim, d_r)
                r_data = cur_generator.get_lsh(samp_n, self.irr_dim)
                n_data = self.normalizer.normalize(r_data)
                p_data = self.normalizer.renormalize([aec.predict(np.array(xx).reshape(1,self.dim))[0] for xx in n_data])
                y_orig = [self.func(x) for x in r_data]
                y_pred = [self.func(x) for x in p_data]
                y_error = mean_absolute_error(y_orig, y_pred)
                err.append(y_error)
          
            ax = fig.add_subplot(1, self.dim, (i+1))
            color = np.random.rand(3)
            ax.bar(names, err, color = color)
            for tick in ax.get_xticklabels():
                tick.set_rotation(35)   
        
        fig.subplots_adjust(wspace = 0.3, bottom=0.3)
        fig.suptitle(f'Mean Y error for each parameter. Mean Y error: {y_mean_error:.3f}', fontsize = 14)
        fig.savefig('../../Saved models/Graphs/' + f'{self.func.name}_{aec.type}_error.png', format = 'png')
        return y_mean_error, fig