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
    
    optimizer = ParamsSelection()
    with suppress_stdout():
        x_opt, y_err = optimizer.ego('dense',func, 60000, 6, 15)
    
    print(f'Opt params:\nepohs = {int(x_opt[0])}\nbatch = {int(x_opt[1])}\nencoded dim = {int(x_opt[2])}\nsample split = {x_opt[3]*100:.2f} % : {(1.0 - x_opt[3])*100:.2f} %')
    print(f'Opt Y error: {y_err}')
    