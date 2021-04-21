import sys
import pathlib
import cProfile
import time

import pandas as pd
import numpy as np

sys.path.append(str(pathlib.Path().absolute()))

from models.kmeans import KMeans
from tests.utils import TestUtils


if __name__ == '__main__':
  model = KMeans(n_clusters=4, iterations=5, logging=True)
  dataset3 = np.loadtxt('data/credit_card_customers/bank_churn.txt') # 11 linhas
  dataset2 = np.loadtxt('data/customer_churn/customer_churn_processed.txt') # 7.043 linhas
  dataset1 = np.loadtxt('data/churn_modelling/test.txt') # 10.000 linhas
  dataset4 = np.loadtxt('data/cateter.txt') # 11 linhas
  fake_data = np.loadtxt('data/fake_data.txt')

  test_utils = TestUtils()

  # prof = cProfile.Profile()
  # prof.enable()
  # prof.run('model.fit(fake_data)')
  # prof.disable()

  # test_utils.write_stats_file(prof)
  #time.perf_counter()