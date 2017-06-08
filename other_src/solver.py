# -*- coding: utf-8 -*-
import configparser
import copy
import datetime
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

class Solver(object):
    def __init__(self,n,m,R,A,b):
        self.x = cvx.Variable(m)
        self.R = R

        self.n = n
        self.A = np.reshape(A,(n*m,m))
        # self.A = np.empty((0,m),float)
        # for i in range(n):
        #     for j in range(m):
        #         np.append(self.A, np.array([A[i][j]]), axis = 0)
        self.b = np.reshape(b,(n*m,1))
        # print(self.A)
        # print(self.b)
        self.problem()

    def problem(self):
        obj = cvx.Minimize(cvx.norm(self.A *self.x - self.b,2)**2)
        constraints = [cvx.norm(self.x, 2) <= self.R]
        self.prob = cvx.Problem(obj,constraints)


    def solve(self):
        pro = self.prob
        # print(cvx.installed_solvers())
        pro.solve(solver = cvx.ECOS_BB,verbose = True)
        print(pro.value)
        return pro.value

class Solver_linear(Solver):
    def __init__(self, n, m, R, A, b):
        self.x = cvx.Variable(m)
        self.R = R

        self.n = n
        self.A = A
        self.b = b
        # self.A = np.reshape(A, (n * m, m))
        # self.A = np.empty((0,m),float)
        # for i in range(n):
        #     for j in range(m):
        #         np.append(self.A, np.array([A[i][j]]), axis = 0)
        # self.b = np.reshape(b, (n * m, 1))
        # print(self.A)
        # print(self.b)
        self.problem()
    def problem(self):
        obj = None
        # obj = cvx.Minimize(cvx.norm(self.A *self.x - self.b,2))
        # for i in range(self.n):
            # print (cvx.norm(self.A[i]*self.x-self.b[i],2))
        obj = [cvx.norm((self.A[i]*self.x)-self.b[i],2) for i in range(self.n)]
        sum_obj = cvx.Minimize(sum(obj))
        constraints = [cvx.norm(self.x, 2) <= self.R]
        self.prob = cvx.Problem(sum_obj,constraints)
