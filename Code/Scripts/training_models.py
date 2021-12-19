from optimize_class import ParamsSelection
from function_class import TestFunctions
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    test = TestFunctions()
    func = test.func_1()
    dim, irr, _, _, _= func.get_params()
    optimizer = ParamsSelection()
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('dense',func, 60000,dim + irr + 1, 100)
    
    print(f'Function 1 dense training\nOpt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'Opt Y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('deep',func, 60000, dim + irr + 1, 100)
    
    print(f'function 1 deep training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('vae',func, 60000, dim + irr + 1, 100)
    
    print(f'function 1 vae training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    
    func = test.func_2()
    dim, irr, _, _, _= func.get_params()
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('dense',func, 60000, dim + irr + 1, 100)
    
    print(f'function 2 dense training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('deep',func, 60000, dim + irr + 1, 100)
    
    print(f'function 2 deep training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('vae',func, 60000, dim + irr + 1, 100)
    
    print(f'function 2 vae training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    
    func = test.func_3()
    dim, irr, _, _, _= func.get_params()
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('dense',func, 60000, dim + irr + 1, 100)
    
    print(f'function 3 dense training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('deep',func, 60000, dim + irr + 1, 100)
    
    print(f'function 3 deep training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('vae',func, 60000, dim + irr + 1, 100)
    
    print(f'function 3 vae training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    #--------------------------------------------------------------------#
    
    func = test.func_6()
    dim, irr, _, _, _= func.get_params()
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('dense',func, 60000, dim + irr + 1, 100)
    
    print(f'function 6 dense training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('deep',func, 60000, dim + irr + 1, 100)
    
    print(f'function 6 deep training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')
    
    #--------------------------------------------------------------------#
    
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('vae',func, 60000, dim + irr + 1, 100)
    
    print(f'function 6 vae training\nopt params:\nepochs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'opt y error: {y_err}\n')