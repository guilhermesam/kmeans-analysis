import cProfile
import pstats
import io
import random
import time
import sys
import pathlib
import os

import numpy as np

sys.path.append(str(pathlib.Path().absolute()))

from models.kmeans import KMeans
from tests.utils import file_length


class KMeansTestCase:
  def __init__(self):
    self.model = None


  def write_stats_file(self, profile):
    s = io.StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('stats.txt', 'w+') as f:
      f.write(s.getvalue())


  def read_info_from_stats_file(self):
    file_path = 'stats.txt'
    HEADER_END_LINE = 4
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    del lines[:HEADER_END_LINE]
    new_file = open(file_path, 'w+')

    for line in lines:
      new_file.write(line)

    new_file.close()


  def __generate_fake_data(self, x_length, y_length):
    dataset = []
    for line in range(y_length):
      data = []
      for point in range(x_length):
        if point % 2 == 0:
          data.append(random.randint(0, 30))
        else:
          data.append(random.randint(1000, 4000))
      dataset.append(data)
    return np.array(dataset)
  

  def __write_execution_time_file(self, time, n_clusters, iterations, file_length):
    filepath = 'performance_results.txt'
    file = open(filepath, 'a')
    file.write('{}\t{}\t{}\t{}\n'.format(time, n_clusters, iterations, file_length))


  def performance_test(self, data, times, model):
    file_len = file_length(data)
    self.model = model

    timers = []

    for execution in range(times):
      start = time.perf_counter()
      self.model.fit(data)
      end = time.perf_counter()
      timers.append(end - start)

    self.__write_execution_time_file(
      sum(timers) / len(timers),
      self.model.n_clusters,
      self.model.iterations,
      file_len
    )

  
  def __add_header_to_performance_file(self):
    file_name = 'performance_results.txt'
    line = 'avg time\tn_clusters\titerations\tfile_length'
    dummy_file = file_name + '.bak'
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        write_obj.write(line + '\n')
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)
    

  def run(self):
    EXECUTION_TIMES = 5

    sample_data_lengths = [1000, 2500, 5000]
    NUM_COLUMNS = 4
    sample_data = [self.__generate_fake_data(NUM_COLUMNS, y) for y in sample_data_lengths]

    # testando com diferentes tamanhos de arquivos
    model = KMeans(n_clusters=4, iterations=50)
    print('testing with different file lengths')

    for data in sample_data:
      print('current: ', data.shape[0])
      self.performance_test(data, EXECUTION_TIMES, model)

    print('testing with different n_clusters')
    print('data size: ', sample_data_lengths[0])
    # testando com diferentes números de clusters
    for n_clusters in [4, 8, 12]:
      print('current: ', n_clusters)
      model = KMeans(n_clusters=n_clusters, iterations=50)
      self.performance_test(sample_data[0], EXECUTION_TIMES, model)

    self.__add_header_to_performance_file()
    print('results saved in performance_results.txt file')


if __name__ == '__main__':
  test = KMeansTestCase()    
  test.run()