# -*- coding: utf-8 -*-
import copy

import numpy as np

from agent_src.agent import Agent_i


class Agent_i_D_NG(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        super(Agent_i_D_NG, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.y_i = copy.copy(self.x_i)
        self.y_j = np.zeros_like(self.x_j)

    def s(self, k):
        return self.step / (k)

    def beta(self, k):
        return (k - 1) / (k + 2)

    def send(self, j):
        return self.y_i

    def receive(self, j, y_j):
        self.y_j[j] = y_j

    def d_i(self):
        A_T = self.A.transpose()
        return 2.0 * np.dot(A_T, (np.dot(self.A, self.y_i) - self.b))

    def koshin(self, k):
        sum = 0.0
        x_i_bf = self.x_i
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.y_j[j] - self.y_i)

        z_i = self.y_i + sum - (self.s(k) * self.d_i())
        self.x_i = z_i
        self.y_i = self.x_i + self.beta(k) * (self.x_i - x_i_bf)


class Agent_i_nedic(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s):
        super(Agent_i_nedic, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.y_i = self.d_i()
        self.y_j = np.zeros_like(self.x_j)

    def s(self):
        return self.step

    def send(self, j):
        return (self.x_i, self.y_i)

    def receive(self, j, x_j):
        self.x_j[j] = x_j[0]
        self.y_j[j] = x_j[1]

    def koshin(self, k):
        sum = 0.0
        d = self.d_i()
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)

        z_i = self.x_i + sum - self.s() * self.y_i
        self.x_i = z_i

        sum = 0.0
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.y_j[j] - self.y_i)
        z_i = self.y_i + sum + self.d_i() - d
        self.y_i = z_i


class Agent_i_harnessing(Agent_i):
    def __init__(self, n, m, weight, system_A, output_b, number, s, R):
        super(Agent_i_harnessing, self).__init__(n, m, weight, system_A, output_b, number, s)
        self.s_i = self.d_i()
        self.s_j = np.zeros_like(self.x_j)

    def s(self):
        return self.step

    def send(self, j):
        return (self.x_i, self.s_i)

    def receive(self, j, x_j):
        self.x_j[j] = x_j[0]
        self.s_j[j] = x_j[1]

    def koshin(self, k):
        sum = 0.0
        sum1 = 0.0
        d_bf = self.d_i()
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)
                sum1 += self.a[j] * (self.s_j[j] - self.s_i)
        z_i = self.x_i + sum - self.s() * self.s_i
        self.x_i = z_i
        self.s_i = self.s_i + sum1 + self.d_i() - d_bf


class Agent_i_harnessing_2(Agent_i_harnessing):
    def koshin(self,k):
        sum = 0.0
        d_bf = self.d_i()
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.x_j[j] - self.x_i)
        z_i = self.x_i + sum - self.s() * self.s_i
        self.x_i = z_i
        self.s_i = self.s_i + self.d_i() - d_bf

class Agent_i_harnessing_event(Agent_i_harnessing):
    def __init__(self, n, m, weight, system_A, output_b, number, s,R):
        super(Agent_i_harnessing_event, self).__init__(n, m, weight, system_A, output_b, number, s,R)
        self.trigger_check = np.ones(n)
        self.tildex_ij = np.zeros((self.n, self.m, 1))
        self.tildex_ji = np.zeros((self.n, self.m, 1))
        self.tildes_ij = np.zeros((self.n, self.m, 1))
        self.tildes_ji = np.zeros((self.n, self.m, 1))
        for i in range(self.n):
            self.tildes_ij[i] = self.d_i()

    def trigger_judge(self, k):
        for j in range(self.n):
            if self.a[j] > 0:
                if np.linalg.norm(self.x_i - self.tildex_ij[j]) >= self.threshold(k,j):
                    self.trigger_check[j] = 1
                else:
                    self.trigger_check[j] = 0

    def send(self, j):
        # if self.graph_trigger_count == self.neighbor_count:
        #     self.graph_trigger_count = 1
        # else:
        #     self.graph_trigger_count += 1

        if self.trigger_check[j] == 1:
            self.tildex_ij[j] = copy.copy(self.x_i)
            self.tildes_ij[j] = copy.copy(self.s_i)
            # self.trigger_count[j] += 1
            # self.trigger_time[j].append(self.graph_trigger_count)
            # self.trigger_time2[j].append(-10)
            return (self.x_i,self.s_i)
        else:
            # self.trigger_time[j].append(-10)
            # self.trigger_time2[j].append(self.graph_trigger_count)
            return None

    def receive(self, j, x_j):
        if x_j is None:
            return
        else:
            self.tildex_ji[j] = x_j[0]
            self.tildes_ji[j] = x_j[1]


    def threshold(self, k,j):
        # if self.th_pattern == 0:
        #     if self.a[j] > 0.15:
        #         return self.s(k) * self.E
        #     else:
        #         return 2*self.s(k)*self.E
        return 0.99**k

    def koshin(self, k):
        sum = 0.0
        sum1 = 0.0
        d_bf = self.d_i()
        for j in range(self.n):
            if j != self.name:
                sum += self.a[j] * (self.tildex_ji[j] - self.tildex_ij[j])
                sum1 += self.a[j] * (self.tildes_ji[j] - self.tildes_ij[j])
        z_i = self.x_i + sum - self.s() * self.s_i
        self.x_i = z_i
        self.s_i = self.s_i + sum1 + self.d_i() - d_bf
        self.trigger_judge(k + 1)

        # elif self.th_pattern == 1:
        #     return 1.0/((k+1))
        #
        # elif self.th_pattern == 2 :
        #     return 10.0/((k+1))
        #
        # elif self.th_pattern == 3:
        #     return 100.0/((k+1))
        #
        # elif self.th_pattern == 4:
        #     return 10.0 / ((k + 1)**2)
        #
        # elif self.th_pattern == 5:
        #     return 10.0 / ((k + 1) ** 0.7)
        #
        # elif self.th_pattern == 6:
        #     return 10.0 / ((k + 1) ** 0.5)
