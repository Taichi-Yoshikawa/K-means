# << Intelligent systems >>
# REPORT#3 : K-means
# - K-means

import os
import time                         as tm
import numpy                        as np
import pandas                       as pd
from matplotlib  import pyplot      as plt
import itertools                    as it
from sklearn  import metrics        as mr
import configuration                as cf


# ---------
#  K-means
# ---------
class KMeans:
    '''
        K-means
    '''
    def __init__(self, seed=1):
        # configuration
        self.seed = seed  if isinstance(seed,(range,list)) else [seed]
        self.cnf = cf.Configuration()
        self.random = np.random
        self.time = tm.strftime('%Y-%m-%d_%H-%M-%S')
        self.cnf_name = 'comparison_' + '_'.join(['Iris',self.cnf.similarity_index])


    def main(self):
        '''
            main function
        '''
        try:
            for i in range(len(self.seed)):
                # set seed-value of numpy.random
                self.random.seed(self.seed[i])
                # loading dataset
                (x, t) = self.loadDataset(i)
                # random initialize
                centroids = x[self.random.choice(range(x.shape[0]), self.cnf.centers, replace=False)]
                # initialize prediction-label by k-means
                t_predict = np.zeros(x.shape[0])
                # label :
                #   0 : no-assignment
                #   n : assign n-cluster (n = 1,2,3,...)
                t_predict_list, centroids_list = [], []
                acc_list, confmat_list = [], []

                for j in range(1, self.cnf.upper_limit_iter + 1):
                    # calculate similarity index
                    similarity = self.calculateSimilarityIndex(x, centroids)
                    # assignment clusters
                    t_predict, movement = self.assignClusters(similarity, t_predict)
                    # move centroids
                    centroids = self.moveCentroids(x, t_predict)
                    # calculate accuracy
                    acc, confmat, t_predict_label, order = self.calculateAccuracy(t, t_predict)
                    # plot figure
                    self.plotFigure(x, t, t_predict_label, centroids, order, i, j)
                    # record
                    t_predict_list.append(t_predict_label)
                    centroids_list.append(centroids)
                    acc_list.append(acc)
                    confmat_list.append(confmat)
                    # movement
                    if movement :
                        print('{} iters : class-move (accuracy: {}[%])'.format(str(j).rjust(3), acc*100))
                    else :
                        print('{} iters : class-stop (accuracy: {}[%])'.format(str(j).rjust(3), acc*100))
                        break

                # save experimental data
                self.saveExperimentalData({'t-pred':np.array(t_predict_list), 'centroid':np.array(centroids_list), 'acc':np.array(acc_list), 'confmat':np.array(confmat_list)},i)

        except Exception as e:
            print('Error : {}'.format(e))


    def calculateSimilarityIndex(self, x, centroids):
        '''
            calculate similarity index (distance)
            arguments:
                x: input-data positions
                centroids: centroid positions
            returns:
                similarity: similarity between input-data positions and centroid positions
                    [[similarity_centroid_1, similarity_centroid_2, ...], ...]
        '''
        similarity = []
        if self.cnf.similarity_index == 'euclidean-distance':
            for i in range(x.shape[0]):
                similarity.append([ np.sum((centroids[j] - x[i])**2)  for j in range(centroids.shape[0]) ])
            similarity = np.array(similarity)
        else :
            print('Error : similarity_index is invalid strings {}'.format(self.cnf.similarity_index))
            return

        return similarity


    def assignClusters(self, similarity, t_predict):
        '''
            assign cluster by minimizing similarity
            arguments:
                similarity: similarity between input-data positions and centroid positions
                t_predict: prediction label
            returns:
                new_t_predict: new prediction label
                movement: whether prediction label move (True/False)
        '''
        new_t_predict = similarity.argmin(axis=1) + 1
        movement = True
        if (new_t_predict == t_predict).all() :
            movement = False

        return new_t_predict, movement


    def moveCentroids(self, x, t_predict):
        '''
            move Centroids
        '''
        centroids = [ x[t_predict == label].mean(axis=0)  for label in range(1, self.cnf.centers+1) ]
        centroids = np.array(centroids)
        return centroids


    def loadDataset(self, i):
        '''
            load dataset from URL
        '''
        if (i==0) and not (os.path.exists(self.cnf.path_out + '/' + self.cnf.dataset_url.split('/')[-1])) :
            df = pd.read_csv(self.cnf.dataset_url, header=None)
            df.to_csv(self.cnf.path_out + '/' + self.cnf.dataset_url.split('/')[-1], header=False, index=False)
        else:
            df = pd.read_csv(self.cnf.path_out + '/' + self.cnf.dataset_url.split('/')[-1], header=None)
        # select [sepal_length, sepal_width, petal_length, petal_width]
        x_all = df[self.cnf.dataset_index['dec']].values
        t_all_origin = df[self.cnf.dataset_index['obj']].values
        # transform one-hot vector
        t_all = []
        for i in range(len(t_all_origin)):
            t_all.append(self.cnf.dataset_one_hot_vector[t_all_origin[i]])
        t_all = np.array(t_all)
        x = x_all[:].copy()
        t = t_all[:].copy()

        return (x, t)


    def plotFigure(self, x, t, t_predict, centroids, order=None, seed=0, iter=None):
        '''
            plot figure
        '''
        folder_name = 'graph'
        path_graph = self.cnf.path_out + '/' + folder_name
        # make directory
        if not os.path.isdir(path_graph):
            os.makedirs(path_graph)

        fig = plt.figure(figsize=(12,6))
        color = ['blue', 'green', 'red']
        color_centroids = ['darkblue', 'darkgreen', 'darkred']
        marker = ['o', '*']

        # graph-1 : True class
        title1 = 'True class'
        ax1 = fig.add_subplot(1,2,1)
        keys = list(self.cnf.dataset_one_hot_vector.keys())
        values = list(self.cnf.dataset_one_hot_vector.values())
        for i in range(self.cnf.centers):
            x_label = x[t == values[i]]
            ax1.scatter(x_label[:,0], x_label[:,1], color=color[i], label=keys[i], linewidth = 1.0, marker=marker[0])
        ax1.set_title(title1)
        ax1.set_xlabel(self.cnf.dataset_dec[0])
        ax1.set_ylabel(self.cnf.dataset_dec[1])
        ax1.legend()

        # graph-2 : True class
        title2 = 'k-means class'
        ax2 = fig.add_subplot(1,2,2)
        for i in range(self.cnf.centers):
            x_label = x[t_predict == (i+1)]
            ax2.scatter(x_label[:,0], x_label[:,1], color=color[i], label='class-'+ str(i+1), linewidth = 1.0, marker=marker[0])
            if order is None:
                ax2.scatter(centroids[i,0], centroids[i,1], color=color_centroids[i], marker=marker[1])
            else:
                ax2.scatter(centroids[i,0], centroids[i,1], color=color_centroids[order[i]-1], marker=marker[1])
        ax2.set_title(title2)
        ax2.set_xlabel(self.cnf.dataset_dec[0])
        ax2.set_ylabel(self.cnf.dataset_dec[1])
        ax2.legend()

        if iter == None:
            fig.savefig(path_graph + '/' + self.cnf_name + '.png' , dpi=300)
            #plt.show()
        elif isinstance(iter, int) :
            fig.savefig(path_graph + '/' + self.cnf_name + '_seed=' + str(self.seed[seed]) + '_iter=' + str(iter) + '.png' , dpi=300)

        plt.close()


    def saveExperimentalData(self, data_dict=None, seed=0):
        '''
            save experimental data
        '''
        if not data_dict is None:
            folder_name = 'table'
            path_table = self.cnf.path_out + '/' + folder_name
            # make directory
            if not os.path.isdir(path_table):
                os.makedirs(path_table)
            data_length = []
            for i in range(len(data_dict)):
                data_length.append(len(list(data_dict.values())[i]))
            max_iter = max(data_length)
            value, index = None, None

            for i in range(len(data_dict)):
                data = data_dict[list(data_dict.keys())[i]]
                if i == 0 :
                    # vector
                    if data.ndim == 1:
                        value = np.array([data]).T
                        index = [list(data_dict.keys())[i]]
                    # matrix
                    elif data.ndim == 2:
                        value = data
                        index = [ '{}[{}]'.format(list(data_dict.keys())[i], j+1)  for j in range(data.shape[1]) ]
                    # tensor
                    else:
                        value = data.reshape([max_iter,-1])
                        shape1, shape2 = np.arange(data.shape[1]), np.arange(data.shape[2])
                        index = [ '{}[{}][{}]'.format(list(data_dict.keys())[i], j+1, k+1) for j,k in zip(shape1.repeat(data.shape[2]),shape2.tile(data.shape[1]))]
                else :
                    if data.ndim == 1:
                        value = np.block([[value, np.array([data]).T]])
                        index.append(list(data_dict.keys())[i])
                    # matrix
                    elif data.ndim == 2:
                        value = np.block([[value, data]])
                        index.extend([ '{}[{}]'.format(list(data_dict.keys())[i], j+1)  for j in range(data.shape[1]) ])
                    # tensor
                    else:
                        value = np.block([[value, data.reshape([max_iter,-1])]])
                        shape1, shape2 = np.arange(data.shape[1]), np.arange(data.shape[2])
                        index.extend([ '{}[{}][{}]'.format(list(data_dict.keys())[i], j+1, k+1) for j,k in zip(np.repeat(shape1,data.shape[2]),np.tile(shape2,data.shape[1]))])

            df = pd.DataFrame(value, index=range(1,max_iter+1),columns=index)
            df.to_csv(path_table + '/' + 'experimental-data_' + self.cnf_name+ '_seed=' + str(self.seed[seed]) + '.csv')


    def calculateAccuracy(self, t, t_predict):
        '''
            calculate accuracy
        '''
        pt = list(it.permutations(list(self.cnf.dataset_one_hot_vector.values()), len(self.cnf.dataset_one_hot_vector)))
        # transform string
        str_t_predict = t_predict.astype(str)
        confmat, acc = [], 0

        for i in range(len(pt)):
            transform_table = { str(j+1):str(pt[i][j])  for j in range(len(pt[i])) }
            t_predict_ = np.array([ transform_table[str_t_predict[j]]  for j in range(str_t_predict.shape[0]) ], dtype=np.int)
            acc_ = round(mr.accuracy_score(t,t_predict_),4)
            if acc_ > acc :
                acc = acc_
                confmat = mr.confusion_matrix(t,t_predict_)
                t_truepredict = t_predict_.copy()
                order = pt[i]

        return acc, confmat, t_truepredict, order


if __name__ == "__main__":
    km = KMeans(range(10))
    km.main()
