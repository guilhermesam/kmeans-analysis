try:
	import argparse
	import logging
	import sys
	import numpy as np

except ImportError as error:
	print(error)
	print()
	print('You must install the requirements:')
	print('  pip3 install --upgrade pip')
	print('  pip3 install -r requirements.txt ')
	print()
	sys.exit(-1)

def a():
    print(1)

class KMeans:
    def __init__(self, n_clusters, iterations, logging=False):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.centroids = {}
        self.logging = logging
        self.classifications = {}

        self.__str__ = 'KMeans(n_clusters={}, iterations={})'.format(self.n_clusters, self.iterations)


    def write_classifications_into_json(self):
        import json
        new_dict = {}
        for key in model.classifications.keys():
            new_dict.update({key: []})
            
        for key in model.classifications.keys():
            for value in model.classifications[key]:
                new_dict[key].append(value.tolist())

        with open('classifications.json', 'w') as file:
            json.dump(new_dict, file)


    def __initialize_random_centroids(self, data):
        random_indexes = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        for i in range(self.n_clusters):
            self.centroids[i] = data[random_indexes[i]]


    def __initialize_dict_of_empty_lists(self, length):
        this_dict = {}
        for i in range(length):
            this_dict[i] = list()
        return this_dict


    def fit(self,data):
        self.__initialize_random_centroids(data)
        EPOCH_CHECKPOINT = 10
        
        for epoch in range(self.iterations):
            if self.logging and epoch % EPOCH_CHECKPOINT == 0:
                print(epoch)

            self.classifications = self.__initialize_dict_of_empty_lists(self.n_clusters)

            for row in data:
                distances_to_centroid = [np.linalg.norm(row - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances_to_centroid.index(min(distances_to_centroid))
                self.classifications[classification].append(row)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        return distances.index(min(distances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help='O caminho do arquivo que contém o conjunto de dados', type=str)
    parser.add_argument('--n_clusters', help='O número de clusters que o modelo deve encontrar', type=int)
    parser.add_argument('--iterations', help='O número de iterações que o modelo deve percorrer', type=int)
    parser.add_argument('--save', help='Salvar classificações em um arquivo JSON')
    parser.add_argument('--log', help='Mostrar o resultado das classificações na tela')

    args = parser.parse_args()
    
    model = KMeans(n_clusters=args.n_clusters, iterations=args.iterations)
    data = np.loadtxt(args.filepath)
    
    model.fit(data)

    if args.save:
        model.write_classifications_into_json()

    if args.log:
        print(model.classifications)