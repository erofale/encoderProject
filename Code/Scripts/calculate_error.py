from autoencoder_class import AutoencoderClass
from function_class import TestFunctions
from error_class import ErrorCalculate
from optimize_class import ParamsSelection
import sys, os
import argparse

def createParser ():
    parser = argparse.ArgumentParser(prog = 'calculate_error',
            description = 'Запуск расчёта ошибки выбранного автоэнкодера для выбранной функции')
    parser.add_argument ('-f', '--func', choices = TestFunctions.get_func_names() + ['all'],
                         default = 'all', type = str, help = 'Название функции')
    parser.add_argument ('-a', '--aec', choices = AutoencoderClass.get_aec_types() + ['all'],
                         default = 'all', type = str, help = 'Тип автоэнкодера')
    return parser

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    f_name = namespace.func
    aec_type = namespace.aec
    
    func_names = TestFunctions.get_func_names()
    aec_types = AutoencoderClass.get_aec_types()
    
    files = os.listdir('../../Saved models/Params/')
    
    if(f_name == 'all'):
        for f in func_names:
            func = TestFunctions.get_func(f)
            dim, irr, _, _, _= func.get_params()
            optimizer = ParamsSelection()

            if(aec_type == 'all'):
                for aec in aec_types:
                    name = f + '_ego' + aec
                    finding = [f for f in files if name in f and f.endswith(".txt")]
                    model = AutoencoderClass.create_from_file(finding[0])      
                    err_calc = ErrorCalculate(func)
                    error, fig = err_calc.calculate(model)
                    print(f'Mean Y error {f} {aec}: {error:.3f}')
            else:
                name = f + '_ego' + aec_type
                finding = [f for f in files if name in f and f.endswith(".txt")]
                model = AutoencoderClass.create_from_file(finding[0])      
                err_calc = ErrorCalculate(func)
                error, fig = err_calc.calculate(model)
                print(f'Mean Y error {f} {aec_type}: {error:.3f}')
    
    else:
        func = TestFunctions.get_func(f_name)
        dim, irr, _, _, _= func.get_params()
        optimizer = ParamsSelection()
        if(aec_type == 'all'):
            for aec in aec_types:
                name = f_name + '_ego' + aec
                finding = [f for f in files if name in f and f.endswith(".txt")]
                model = AutoencoderClass.create_from_file(finding[0])      
                err_calc = ErrorCalculate(func)
                error, fig = err_calc.calculate(model)
                print(f'Mean Y error {f_name} {aec}: {error:.3f}')
        else:
            name = f_name + '_ego' + aec_type
            finding = [f for f in files if name in f and f.endswith(".txt")]
            model = AutoencoderClass.create_from_file(finding[0])      
            err_calc = ErrorCalculate(func)
            error, fig = err_calc.calculate(model)
            print(f'Mean Y error {f_name} {aec_type}: {error:.3f}')
