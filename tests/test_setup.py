import cProfile
import pstats
import io
import random
import time
import sys, getopt
import pathlib
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipy_opt

sys.path.append(str(pathlib.Path().absolute()))

from models.kmeans import KMeans
from tests.utils import file_length, funcao_fatorial, funcao_linear, funcao_quadratica


class KMeansTestCase:
  def __init__(self):
    self.model = None
    self.tamanhos = [5000, 7500, 10000, 12500, 15000]
    self.medias = [5, 7, 12, 16, 20]

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
  
  def __write_execution_time_file(self, output, time, n_clusters, iterations, file_length):
    f = open(output, 'a')
    f.write('{}\t{}\t{}\t{}\n'.format(time, n_clusters, iterations, file_length))

  def performance_test(self, output, data, times, model):
    file_len = file_length(data)
    self.model = model

    timers = []

    for execution in range(times):
      start = time.perf_counter()
      self.model.fit(data)
      end = time.perf_counter()
      timers.append(end - start)

    self.__write_execution_time_file(
      output,
      sum(timers) / len(timers),
      self.model.n_clusters,
      self.model.iterations,
      file_len
    )

  def __add_header_to_performance_file(self):
    file_name = 'performance_results.txt'
    line = 'avg_time\tn_clusters\titerations\tfile_length'
    dummy_file = file_name + '.bak'
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        write_obj.write(line + '\n')
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)
    
  def run(self, argv,charts = False ):
    try:
      opts, args = getopt.getopt(argv,"n:",["name="])
    except getopt.GetoptError:
      print('Utilize o argumento de nome:\n  python3 teste_setup -n <name>')
      sys.exit(2)
    if(opts==[] or opts[0][0]!="-n" or len(opts)!=1):
      print ('Utilize o argumento de nome corretamente:\n  python3 teste_setup -n <name>')
      sys.exit()
    
    name = opts[0][1]
    
    PERSON = name
    
    OUTPUTS = ['out/{}_results_length.txt'.format(PERSON), 'out/{}_results_nclusters.txt'.format(PERSON)]
    for fname in OUTPUTS:
      open(fname, 'w').close()
      
    EXECUTION_TIMES = 1

    NUM_COLUMNS = 2
    sample_data = [self.__generate_fake_data(NUM_COLUMNS, y) for y in self.tamanhos]

    # testando com diferentes tamanhos de arquivos
    model = KMeans(n_clusters=4, iterations=50)
    print('testing with different file lengths')

    for data in sample_data:
      print('current: ', data.shape[0])
      self.performance_test(OUTPUTS[0], data, EXECUTION_TIMES, model)

    print('testing with different n_clusters')
    print('data size: ', self.tamanhos[0])
    # testando com diferentes números de clusters
    for n_clusters in [4, 6, 8, 10, 12]:
      print('current: ', n_clusters)
      model = KMeans(n_clusters=n_clusters, iterations=50)
      self.performance_test(OUTPUTS[1], sample_data[0], EXECUTION_TIMES, model)

    if (charts):
      for fname in OUTPUTS:
        f = np.loadtxt(fname).T
        fig, ax = plt.subplots()

        if 'length' in fname:
          parametros, pcov = scipy_opt.curve_fit(funcao_linear, xdata=self.tamanhos, ydata=f[0])
          self.aproximados = [funcao_linear(x, *parametros) for x in self.tamanhos]
          print("aproximados:           {}".format(self.aproximados))
          print("parametros_otimizados: {}".format(parametros))
          print("pcov:                  {}".format(pcov))
          print()

          plt.suptitle("Tempo X Tamanho do dataset ({})\n".format(PERSON), fontsize=14)
          plt.title("Iterações: {} N Clusters: {}".format(int(f[2][0]), int(f[1][0])), fontsize=10)
          plt.plot(f[3], f[0], '-bo', color='blue', mfc='b', mec='b', markersize=4, linewidth=2)
          plt.xlim([min(f[3]), max(f[3])])
          plt.xlabel("Tamanho")
          plt.xticks(f[3])
          plt.ylim([0, 60])
          plt.ylabel("Tempo (s)")
          plt.plot(self.tamanhos, self.aproximados, '-o', markersize=4, label="Teste", color="red")
          plt.legend(["k-means medido", "k-means aproximado"])
          plt.grid(linestyle='dotted')
          plt.savefig(fname[:-4] + '.png')
          # plt.yticks(f[0])

        else:
          self.tamanhos = [4, 6, 8, 10, 12]
          parametros, pcov = scipy_opt.curve_fit(funcao_linear, xdata=self.tamanhos, ydata=f[0])
          self.aproximados = [funcao_linear(x, *parametros) for x in self.tamanhos]
          print("aproximados:           {}".format(self.aproximados))
          print("parametros_otimizados: {}".format(parametros))
          print("pcov:                  {}".format(pcov))
          print()
          plt.suptitle("Tempo X Quantidade de clusters ({})\n".format(PERSON), fontsize=14)
          plt.title("Iterações: {} Tamanho dataset: {}".format(int(f[2][0]), int(f[3][0])), fontsize=10)
          plt.plot(f[1], f[0], '-mo', color='purple', mfc='m', mec='m', markersize=4, linewidth=2)
          plt.xlabel("N Clusters")
          plt.xticks([4, 6, 8, 10, 12])
          plt.xlim([min(f[1]), max(f[1])])
          plt.ylim([0, 60])
          plt.ylabel("Tempo (s)")
          plt.plot([4, 6, 8, 10, 12], self.aproximados, '-o', markersize=4, label="Teste", color="red")
          plt.legend(["k-means medido", "k-means aproximado"])
          plt.grid(linestyle='dotted')
          plt.savefig(fname[:-4] + '.png')
       
          #plt.yticks(f[0])          
        
        self.medias = f[0]
        # plt.plot(parametros, f[0], width=1, color="red")


if __name__ == '__main__':
  #Argumentos na linha de comando:
  # python3 teste_setup -n "i5-6600K CPU @ 3.50GHz"
  
  test = KMeansTestCase()    
  test.run(sys.argv[1:],charts = True)