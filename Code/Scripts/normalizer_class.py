from typing import Tuple, List


''' Класс нормировки данных '''
class Normalizer():
    
    '''
    Parameters
    ----------
    dim : int
        Размерность входных данных.
    irr_dim : int
        Количество незначащих переменных.
    range_val_pairs : List[Tuple[float,float]]
        Диапазон значений входных данных.
    norm_min : float, optional
        Нижняя граница нормированных данных (по умолчанию 0).
    norm_max : float, optional
        Верхняя граница нормированных данных (по умолчанию 1).
    '''
    def __init__(self, dim : int, irr_dim : int, range_val_pairs : List[Tuple[float,float]], norm_min : float = 0., norm_max : float = 1.):
        self.dim = dim
        self.irr_dim = irr_dim
        self.range_val_pairs = range_val_pairs
        self.range_val = [(i[1] - i[0]) for i in self.range_val_pairs]
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.range_norm = norm_max - norm_min
    
    
    def __normire(self, entryVal : float, ind : int) -> float:
        '''
        Parameters
        ----------
        entryVal : float
            Входное значение для нормировки.
        ind : int
            Индекс элемента в массиве.

        Returns
        -------
        float
            Нормированное значение.

        '''
        return self.norm_min + ((entryVal - self.range_val_pairs[ind][0]) * self.range_norm / self.range_val[ind])
    
    def __renormire(self, normVal : float, ind : int) -> float:
        '''
        Parameters
        ----------
        normVal : float
            Нормированное значение для денормировки.
        ind : int
            Индекс элемента в массиве.

        Returns
        -------
        float
            Денормированное значение.
        '''
        return self.range_val_pairs[ind][0] + ((normVal - self.norm_min) / self.range_norm * self.range_val[ind])
    
    def normalize(self, data) -> List[List[float]]:
        '''
        Parameters
        ----------
        data : list, array
            Входной набор данных для нормировки.

        Returns
        -------
        List[List[float]]
            Нормированный набор данных.
        '''
        count = 0
        if type(data) == list: 
          count = len(data)
        else:
          count = data.shape[0]
          if count == None:
            count = 1
        normData = []
        for i in range(count):
            cur_sample = []
            for j in range(self.dim):
                cur_sample.append(self.__normire(data[i][j], j))
            for j in range(self.dim, self.dim + self.irr_dim):
                cur_sample.append(data[i][j])
            normData.append(cur_sample)
        return normData
    
    def renormalize(self, normData) -> List[List[float]]:
        '''
        Parameters
        ----------
        normData : list, array
            Нормированный набор данных.

        Returns
        -------
        List[List[float]]
            Денормированный набор данных.
        '''
        count = 0
        if type(normData) == list: 
          count = len(normData)
        else:
          count = normData.shape[0]
          if count == None:
            count = 1
        data = []
        for i in range(count):
            cur_sample = []
            for j in range(self.dim):
                cur_sample.append(self.__renormire(normData[i][j], j))
            for j in range(self.dim, self.dim + self.irr_dim):
                cur_sample.append(normData[i][j])
            data.append(cur_sample)
        return data
