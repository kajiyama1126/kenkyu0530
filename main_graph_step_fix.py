from main_tac_class_step_fix import Program
from main_tac_class_step_fix import Program_linear
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('/Users/kajiyama/PycharmProjects/kenkyu0525/inifile.txt')

problem = str(config['problem']['problem'])#2次関数か線形関数
if problem == 'quadratic':
    program = Program()
elif problem == 'linear':
    program = Program_linear()
program.simulate()