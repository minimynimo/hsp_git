#!python3
"""
2019/8/29
This is an numpy implementation of echo state network (reservoir computing).
"""
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
#from progressbar import ProgressBar

class ESN():
    """
    The Echo State Network.

    args:
     - input_size           : size of the observed model space.
     - output_size          : size of the whole model space. Identical to input_size, if not designated.
     - reservoir_size       : size of the reservoir. Default 1000.
     - adjacency_density    : The density of the adjacency matrix for reservoir evolution. Default 0.5
     - spectral_radius      : Maximum magnitude ofm eigenvalues of adjacency matrix. Default 1.0
     - input_scale          : scale of the input-reservoir mapping matrix. Default 1.0

    Parameters:
     - A
     - W_in
     - W_out
     - reservoir_size
     - input_size
     - output_size
    """

    def __init__(self, input_size, output_size=None, reservoir_size=1000, adjacency_density=0.5, spectral_radius=1.0, input_scale=1.0):
        #print('model init')
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        if output_size == None:
            self.output_size = input_size
        else:
            self.output_size = output_size

        self.A = sparse.rand(reservoir_size, reservoir_size, density=adjacency_density)
        self.A = np.array(self.A.todense())
        max_eigval = np.abs(LA.eigvals(self.A)).max()
        if max_eigval > 0:
            self.A = self.A * spectral_radius / max_eigval

        #self.A = self.A * spectral_radius / (np.abs(LA.eigvals(self.A)).max())

        self.W_in = np.zeros((reservoir_size, input_size))
        q = int(reservoir_size / input_size)
        for i in range(input_size):
            np.random.seed(seed=i)
            self.W_in[i*q:(i+1)*q, i] = input_scale * (np.random.rand(q) * 2 - 1)

    def W_in_val(self):
        return self.W_in

    def A_val(self):
        return self.A

    def forward(self, input_vector, previous_state=None):
        """
        Compute the reservoir states of next timestep.

        args:
            - input_vector      : model state at current timestep.
            - previous_state    : reservoir state at current timestep.
        """
        if previous_state is None:
            previous_state = np.zeros((self.reservoir_size, 1))

        # reshape vectors to be column vectors.
        input_vector = input_vector.copy().reshape(-1, 1)
        previous_state = previous_state.copy().reshape(-1, 1)

        #print(input_vector.shape) # 4,1
        # compute the reservoir state at next timestep.
        r = np.tanh(np.dot(self.A, previous_state) + np.dot(self.W_in, input_vector))
        #           300*300 @ 300,1             + 300*4 @ 4,1

        # returns the reservoir state vector as a row vector.
        return r.reshape(-1).copy()


    def train(self, observed_data, target_data=None, washout=0, ridge_param=0.01, spinup = False):
        """
        Train the network with input data.
        args:
        - observed_data     : input matrix (npoints * length). Each columns stands for the state at each time.
        - target_data       : output matrix. Designate if input data (observed data) is imperfect.
        - washout           : timesteps to drop.
        - ridge_param       : parameter for ridge regression
        """
        # observed_data <- input_cal    # (4,n)
        # target_data <- target_cal     # (n, )

        if spinup == False:
            spinup = 1
        observed_data = np.tile(observed_data, (1, spinup))
        target_data = np.tile(target_data, spinup)
        target_data = target_data.flatten()
        #print(observed_data.shape)
        #print(target_data.shape)

        if type(target_data) == type(None):
            target_data = observed_data[:, 1:]
            observed_data = observed_data[:, :-1]   

        # arguments assertion
        if observed_data.shape[0] != self.input_size:
            raise ValueError('Observed data size is set to {}, not {}'.format(self.input_size, observed_data.shape[0]))
        #if target_data.shape[0] != self.output_size:
        #    raise ValueError('Target data size is set to {}, not {}'.format(self.output_size, target_data.shape[0]))
        if observed_data.shape[1] != len(target_data):
            raise ValueError('Observed and target data duration should be the same, got {} and {}'.format(observed_data.shape[1], len(target_data)))

        # calculate reservoir state
        train_length = observed_data.shape[1] # n
        #p = ProgressBar(0, train_length)

        reservoir = np.zeros((self.reservoir_size, train_length))   # (R,n)
        reservoir[:, 0] = self.forward(observed_data[:, 0])         # (R, )

        for i in range(1, train_length):
            #p.update(i) # progressbar
            reservoir[:, i] = self.forward(observed_data[:, i], reservoir[:, i-1])
        #p.update(train_length)
        #p.finish()
        
        reservoir_last = reservoir[:, -1].copy()    # (R, )

        # nonlinear transformation
        reservoir = self.nonlinear_transf(reservoir, inplace=True)  # (R,n)

        # washout and target data setting
        reservoir = reservoir[:, washout:]
        target = target_data[washout:]      # (n, )
        target = target.reshape(-1, 1).T    # (1,n)

        # ridge regression
        U = reservoir @ reservoir.T + ridge_param * np.eye(self.reservoir_size) # (R,R)
        Uinv = LA.inv(U)            # (R,R)

        self.W_out = (np.dot(Uinv , np.dot(reservoir, target.T ))).T  # {(R,R) @ [(R*n) @ (n,1)]}.T     # (1,R)  
        return self.W_out, reservoir_last
      

    def predict(self, current_reservoir, test_observed, ptb_func=None, ptb_scale=0.0, nexttime=False, extended_interval=0):
        """
        Generate the predict time series.
        args:
        - current_reservoir     : last reservoir state in training phase
        - test_observed         : whole test data # 4inputs  by obs duration
        - nexttime              : whether to give true state every step
        - extended_interval     : interval to give the true state
        """
        # current_reservoir <- reservoir    #(1,R)
        # test_observed <- input_eva        #(4,m)

        # pertubation
        if ptb_func == 'normal':
            ptb = lambda: np.random.normal(loc=0, scale=ptb_scale, size=self.input_size)
        elif type(ptb_func) != type(None):
            ptb = ptb_func
        else:
            ptb = lambda: np.zeros((self.input_size))

        test_length = test_observed.shape[1]  # m
    
        predict = np.zeros((self.output_size, test_length)) # (1,m)

        #p = ProgressBar(0, test_length)

        # initial predict
        current_reservoir = self.forward(test_observed[:, 0] + ptb(), current_reservoir)    # (R, )
        reservoir_transf = self.nonlinear_transf(current_reservoir, inplace=False)          # (R, )
        predict[:,0] = (self.W_out @ reservoir_transf.reshape(-1,1)).reshape(-1)            # (1, )

        # predict iteration
        for i in range(1, test_length):  # 1,1090
            #p.update(i) # progressbar
            if nexttime:
                current_reservoir = self.forward(test_observed[:, i] + ptb(), current_reservoir) # (R,1)
            elif extended_interval > 0 and i % extended_interval == 0:
                current_reservoir = self.forward(test_observed[:, i-100])
                for j in range(99, 0, -1):
                    current_reservoir = self.forward(test_observed[:, i-j], current_reservoir)
                current_reservoir = self.forward(test_observed[:, i] + ptb(), current_reservoir)
            else:
                current_reservoir = self.forward(predict[:,i-1], current_reservoir) # should be (R,1)
            reservoir_transf = self.nonlinear_transf(current_reservoir, inplace=False) #should be (R,1)
            predict[:,i] = (self.W_out @ reservoir_transf.reshape(-1,1)).reshape(-1)
            #   { (1,R) @ (R,1) }.reshape(-1) should be

            #if predict[:,i] < 0:
            #    predict[:,i] = 0
        #p.update(test_length)
        #p.finish()

        # predict (1,m)
        return predict

    def nonlinear_transf(self, matrix, inplace=False):
        """
        Apply nonlinear row transformation to input matrix.
        """
        if not inplace:
            matrix = matrix.copy()
        row_pre = matrix[0].copy()
        for i in range(2, matrix.shape[0], 2):
            row_tmp = matrix[i].copy()
            matrix[i] = (matrix[i-1] * row_pre).copy()
            row_pre = row_tmp.copy()
        return matrix
