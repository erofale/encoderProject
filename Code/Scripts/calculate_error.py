from autoencoder_class import AutoencoderClass
from function_class import TestFunctions
from error_class import ErrorCalculate
import os

if __name__ == '__main__':
  test = TestFunctions()
  
  func = test.func_1()
  files = os.listdir('../../Saved models/Params/')
  print(files)
  name = func.func_name + '_ego'
  finding = [f for f in files if name in f and f.endswith(".txt")]
  print(finding)
  model = AutoencoderClass.create_from_file(finding[0])

  err_calc = ErrorCalculate(func)

  error, fig = err_calc.calculate(model)
  print(f'Mean Y error: {error:.3f}')