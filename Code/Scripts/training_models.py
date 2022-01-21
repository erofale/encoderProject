from optimize_class import ParamsSelection
from function_class import TestFunctions
from autoencoder_class import AutoencoderClass
from contextlib import contextmanager
import sys, os
import argparse

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

 
def createParser ():
    parser = argparse.ArgumentParser(prog = 'training_models',
            description = 'Запуск подбора параметров выбранного автоэнкодера для выбранной функции')
    parser.add_argument ('-f', '--func', nargs='?', choices = TestFunctions.get_func_names() + ['all'],
                         default = 'all', type = str, help = 'Название функции')
    parser.add_argument ('-a', '--aec', nargs='?', choices = AutoencoderClass.get_aec_types() + ['all'],
                         default = 'all', type = str, help = 'Тип автоэнкодера')
    parser.add_argument ('-i', '--iter', nargs='?',
                         default = 25, type = int, help = 'Количество эпох подбора гиперпараметров автоэнкодера')
    return parser


if __name__ == "__main__":   
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    f_name = namespace.func
    aec_type = namespace.aec
    it = namespace.iter
    
    func_names = TestFunctions.get_func_names()
    aec_types = AutoencoderClass.get_aec_types()
    
    if(f_name == 'all'):
        for f in func_names:
            func = TestFunctions.get_func(f)
            dim, irr, _, _, _= func.get_params()
            optimizer = ParamsSelection()

            if(aec_type == 'all'):
                for aec in aec_types:
                    with suppress_stdout():
                        x_opt, y_err = optimizer.ego(aec ,func, 60000, dim + 1, it)
                    print(f'{f} {aec} training\nOpt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
                    print(f'Opt mean Y error: {y_err}\n')
            else:
                with suppress_stdout():
                    x_opt, y_err = optimizer.ego(aec_type ,func, 60000, dim + 1, it)
                print(f'{f} {aec_type} training\nOpt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
                print(f'Opt mean Y error: {y_err}\n')
    
    else:
        func = TestFunctions.get_func(f_name)
        dim, irr, _, _, _= func.get_params()
        optimizer = ParamsSelection()
        if(aec_type == 'all'):
            for aec in aec_types:
                with suppress_stdout():
                    x_opt, y_err = optimizer.ego(aec ,func, 60000,dim + 1, it)
                print(f'{f_name} {aec} training\nOpt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
                print(f'Opt mean Y error: {y_err}\n')
        else:
            with suppress_stdout():
                x_opt, y_err = optimizer.ego(aec_type ,func, 60000,dim + 1, it)
            print(f'{f_name} {aec_type} training\nOpt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
            print(f'Opt mean Y error: {y_err}\n')
