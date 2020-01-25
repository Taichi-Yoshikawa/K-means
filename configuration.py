# << Intelligent systems >>
# REPORT#3 : K-means
# - Configuration


class Configuration():
    '''
        Configuration
    '''
    def __init__(self):
        # ----- k-means components -----
        ## cluster numbers
        self.centers            = 3
        ## upper limit of iterations
        self.upper_limit_iter   = 1000

        # ----- k-means options -----
        self.similarity_index   = 'euclidean-distance'

        # ----- Dataset Configuration -----
        self.dataset_url    = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        self.dataset_index  = {
            'dec'   : [0,1],
            'obj'   : 4
        }
        self.dataset_dec = ['Sepal Length', 'Sepal Width']
        self.dataset_one_hot_vector = {
            'Iris-setosa'       : [1,0,0],
            'Iris-versicolor'   : [0,1,0],
            'Iris-virginica'    : [0,0,1]
        }

        # ----- I/O Configuration -----
        self.path_out = '.'
