from autoencoder_class import AutoencoderClass
from normalizer_class import Normalizer
from generator_class import DataGenerator
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def func(x):
  return x[0]*x[1] + x[1]*x[2] + x[3]*x[3]


def compare(orig_data, pred_data):
    # clf = svm.SVC(kernel='linear', C=1, random_state=42)
    # scores_x = cross_val_score(clf, orig_data[0:10], pred_data[0:10], cv=5)
    # scores_y = cross_val_score(clf, [func(x) for x in orig_data][0:10], [func(x) for x in pred_data][0:10], cv=5)
    
    y_error = mean_squared_error(orig_data, pred_data)
    x_error = mean_squared_error([func(x) for x in orig_data], [func(x) for x in pred_data])
    
    print(f'Error X: {x_error:.2f}')
    print(f'Error Y: {y_error:.2f}')


if __name__ == "__main__":
    dim = 4
    irr_dim = 2
    data_range = [(0, 100), (0, 100), (0, 100), (0, 100)]
    generator = DataGenerator(dim, data_range)
    normalizer = Normalizer(dim, data_range)
    
    n = 30000
    sobol_data = generator.get_sobol(n, irr_dim)
    data_train = np.array(sobol_data[0:int(n * 0.7)])
    data_test = np.array(sobol_data[int(n * 0.7):n])
    
    model = AutoencoderClass(func, dim + irr_dim, 4, list(['relu', 'sigmoid']), 'dense', normalizer)
    model.fit(data_train, data_test, 30, 16, True)
    
    pred_data = normalizer.renormalize([model.predict(x.reshape(1,dim + irr_dim))[0] for x in sobol_data[0:100]])
    compare(normalizer.renormalize(sobol_data)[0:100], pred_data[0:100])
    
    