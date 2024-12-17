import unittest
import numpy as np
from RLMSO2 import Population, Individual,RLMSO

class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.lb = np.array([0, 0])
        self.ub = np.array([1, 1])
        self.decs = np.array([[0.5, 0.5], [0.6, 0.6]])
        self.objs = np.array([0.75, 0.8])
        self.csts = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.pop = Population(self.decs, self.objs, self.csts, self.lb, self.ub)

    def test_init(self):
        self.assertEqual(self.pop.popSize, 2)
        self.assertTrue(np.array_equal(self.pop.decs, self.decs))
        self.assertTrue(np.array_equal(self.pop.objs, self.objs))
        self.assertTrue(np.array_equal(self.pop.csts, self.csts))
        self.assertTrue(np.array_equal(self.pop.feasible, [False, False]))

    def test_Cal_Constrain(self):
        self.pop.Cal_Constrain()
        self.assertEqual(self.pop.cstNum, 2)
        self.assertTrue(np.array_equal(self.pop.cstViolationFlag, [[True, True], [True, True]]))
        self.assertTrue(np.array_equal(self.pop.cstViolationNum, [2, 2]))
        self.assertTrue(np.array_equal(self.pop.feasible, [False, False]))

    # def test_addInd(self):
    #     new_ind = Individual(dec=np.array([0.7, 0.7]), obj=0.9, cst=np.array([0.5, 0.6]), idx=2)
    #     # self.pop.addInd(new_ind)
    #     self.assertEqual(self.pop.popSize, 3)
    #     self.assertTrue(np.array_equal(self.pop.decs, np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])))
    #     self.assertTrue(np.array_equal(self.pop.objs, np.array([0.75, 0.8, 0.9])))
    #     self.assertTrue(np.array_equal(self.pop.csts, np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])))
    #     self.assertTrue(np.array_equal(self.pop.feasible, [True, True, True]))
import random
if __name__ == '__main__':

    matrix = np.random.normal(loc=0, scale=5, size=(3, 2))

    # 将所有值裁剪到[-10, 10]范围内
    matrix = np.clip(matrix, -10, 10)
    rlmso = RLMSO()
    matrix_binary = rlmso.binary(matrix)
    print(matrix)
    print(matrix_binary)
