import h5py, pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import convolve1d
import os, urllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge as RR
from sklearn.metrics import r2_score
plt.switch_backend('agg')
from numpy.linalg import inv as inv #Used in kalman filter
import sys
sys.path.append('/home/renyi/gwx/hmk/DynamicalComponentsAnalysis-main')

class KalmanFilterRegression(object):

    """
    Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.
    """

    def __init__(self,C=1):
        self.C=C


    def fit(self,X_kf_train,y_train):

        """
        Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_train.T)
        Z=np.matrix(X_kf_train.T)

        #number of time bins
        nt=X.shape[1]

        #Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        #In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        #Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        #In our case, this is the transformation from kinematics to spikes
        H = Z*X.T*(inv(X*X.T)) #Measurement matrix
        Q = ((Z - H*X)*((Z - H*X).T)) / nt #Covariance of measurement matrix
        params=[A,W,H,Q]
        self.model=params
        

    def predict(self,X_kf_test,y_test):

        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        #Extract parameters
        A,W,H,Q=self.model

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)

        #Initializations
        num_states=X.shape[0] #Dimensionality of the state
        states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0] #Initial state
        states[:,0]=np.copy(np.squeeze(state))
        #Get predicted state for every time bin
        from tqdm import tqdm
        for t in tqdm(range(X.shape[1]-1), desc='fit'):
            #Do first part of state update - based on transition matrix
            P_m=A*P*A.T+W
            state_m=A*state

            #Do second part of state update - based on measurement matrix
            K=P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            P=(np.matrix(np.eye(num_states))-K*H)*P_m
            state=state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1]=np.squeeze(state) #Record state at the timestep
        y_test_predicted=states.T
        return y_test_predicted

class LSTMRegression(object):

    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=512,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        import keras
        keras_v1=int(keras.__version__[0])<=1
        from keras.layers import Dense, LSTM, Dropout
        from keras.models import Sequential

        model=Sequential() #Declare model
        #Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        else:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        if keras_v1:
            model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        else:
            model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted


def load_data(filename, high_pass=True, sqrt=True, thresh=5000, zscore_pos=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            min_t = t[0]
            binned_spikes = np.zeros((len(t), d), dtype=np.float32)
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times <= max_t]
                    spike_times = spike_times[spike_times >= min_t]
                    # make sure to ignore the hash here...
                    binned_spikes[:len(spike_times), chan_idx * n_sorted_units + unit_idx - 1] = spike_times
            binned_spikes = binned_spikes[:, np.count_nonzero(binned_spikes,axis=0) > thresh]
            
            result[region] = binned_spikes
        # Find when target pos changes
        target_pos = f["target_pos"][:].T
        target_pos = pd.DataFrame(target_pos)
        has_change = target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
        time = pd.DataFrame(f['t'][0, :].T)
        # Add start and end times to trial info
        change_times = time.index[has_change]
        start_times = change_times[:-1]
        end_times = change_times[1:]
        # Get target position per trial
        temp_target_pos = target_pos.loc[start_times].to_numpy().tolist()
        # Compute reach distance and angle
        reach_dist = target_pos.loc[end_times - 1].to_numpy() - target_pos.loc[start_times - 1].to_numpy()
        reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
        
        # Create trial info
        result['time'] = time.to_numpy()
        result['start_times'] = start_times.to_numpy()
        result['end_times'] = end_times.to_numpy()
        result['target_pos'] = temp_target_pos
        result['reach_dist_x'] = reach_dist[:, 0]
        result['reach_dist_y'] = reach_dist[:, 1]
        result['reach_angle'] = reach_angle
    
        return result

def plot_tuning_curve(data, cell=0):
    X = data['M1'].T 
    reach_angle = data['reach_angle']
    start_times = data['start_times']
    end_times = data['end_times']
    time = data['time']
    t_min = time[0]
    pi = np.pi
    n_bins = int(120)
    binned_spikes = np.zeros(n_bins, dtype=np.float32)
    x = []
    for i in range(n_bins):
        if i==n_bins/2:
            start_angle1 = -(360./n_bins/2) + 180.
            end_angle1 = (360./n_bins/2) - 180.
            index = np.where(((start_angle1 < reach_angle) & (180. >= reach_angle)) | ((-180. < reach_angle) & (end_angle1 >= reach_angle)))
            x.append(i*pi*2/n_bins)
        elif i < n_bins/2:
            start_angle = -(360./n_bins/2) + i * 360./n_bins
            end_angle = (360./n_bins/2) + i * 360./n_bins
            index = np.where((start_angle < reach_angle) & (end_angle >= reach_angle))
            x.append(i*pi*2/n_bins)
        else:
            start_angle = -360. + i * 360./n_bins - (360./n_bins/2)
            end_angle = -360. + i * 360./n_bins + (360./n_bins/2)
            index = np.where((start_angle < reach_angle) & (end_angle >= reach_angle))
            x.append(i*pi*2/n_bins - 2*pi)

        start_time = start_times[index]
        end_time = end_times[index]

        raster = []
        for temp_i in range(len(index[0])):
            temp_raster = X[cell]
            start_timestamp = start_time[temp_i] * 0.004 + t_min 
            end = end_time[temp_i] if (end_time[temp_i] - start_time[temp_i]) < 1000 else (1000 + start_time[temp_i])
            end_timestamp = end * 0.004 + t_min
            temp_raster = temp_raster[start_timestamp < temp_raster]
            temp_raster = temp_raster[temp_raster < end_timestamp]
            binned_spikes[i] += temp_raster.shape[0]
        if len(index[0]) != 0:    
            binned_spikes[i] /= len(index[0])
        else:
            binned_spikes[i] = 0
    x =np.array(x)
    index = np.where(binned_spikes>0)
    y = binned_spikes[index]
    x = x[index]
    def target_func(x, a0, a2, a3):
        return a0 * np.sin(x + a2) + a3
    # a0*sin(a1*x+a2)+a3
    import scipy.optimize as optimize
    a0 = (max(y) - min(y)) / 2
    max_index = y.tolist().index(max(y))
    a2 = pi/2 - x[max_index]
    a3 = a0
    p0 = [a0, a2, a3]
    para, _ = optimize.curve_fit(target_func, x, y, p0=p0)
    print(para)
    y_fit = [target_func(a, *para) for a in x]
    #Get metric of fit
    y_mean=np.mean(y)
    R2=1-np.sum((y_fit-y)**2)/np.sum((y-y_mean)**2)
    print('R2s:', R2)

    # plt.figure() #初始化一张图
    # plt.scatter(x, y, c='red', label='function')
    # x_plot = np.arange(-2 * pi, 2 * pi, pi/10)        #从1到9，间隔1取点
    # y_plot = [target_func(a, *para) for a in x_plot]
    # plt.plot(x_plot, y_plot, c="orange", label="Fitting Line") 

    # # 3.展示图形
    # plt.legend()  # 显示图例
    # plt.xlabel('angle')
    # plt.ylabel('trail') 
    # plt.title(f'{cell}_tuning_curve') 
    # plt.savefig(f"./results/{cell}_tuning_curve.jpg")

    return R2


def plot_raster(data, direction):
    dir2an = {'right': 0.,
              'left': 180.,
              'up': 90.,
              'down': -90.,
    }
    angle = dir2an[direction]
    X = data['M1'].T 
    reach_angle = data['reach_angle']
    start_times = data['start_times']
    end_times = data['end_times']
    time = data['time']
    t_min = time[0]
        
    index = np.where(reach_angle == angle)
    start_times = start_times[index]
    end_times = end_times[index]
    raster = []
    for i in range(len(index[0])):
        temp_raster = X[9]
        start_timestamp = start_times[i] * 0.004 + t_min 
        end_timestamp = end_times[i] * 0.004 + t_min
        temp_raster = temp_raster[start_timestamp - 20 *0.004 < temp_raster]
        temp_raster = temp_raster[temp_raster < end_timestamp]
        trans_raster = temp_raster - start_timestamp
        if trans_raster.shape[0] > 0:
            raster.append(trans_raster)

    plt.eventplot(raster[:][:100])
    plt.xlabel('Time (s)')
    plt.ylabel('trail')
    plt.savefig(f"./{direction}_raster.jpg")

def plot_psth(data, direction, bin_width_s=0.04):
    dir2an = {'right': 0.,
              'left': 180.,
              'up': 90.,
              'down': -90.,
    }
    angle = dir2an[direction]
    X = data['M1'].T 
    reach_angle = data['reach_angle']
    start_times = data['start_times']
    end_times = data['end_times']
    time = data['time']
    t_min = time[0]
        
    index = np.where(reach_angle == angle)
    start_times = start_times[index]
    end_times = end_times[index]
    raster = []
    max = 0
    min = 2
    for i in range(len(index[0])):
        temp_raster = X[9]
        start_timestamp = start_times[i] * 0.004 + t_min 
        end_timestamp = end_times[i] * 0.004 + t_min
        temp_raster = temp_raster[start_timestamp - 20 *0.004 < temp_raster]
        temp_raster = temp_raster[temp_raster < end_timestamp]
        trans_raster = temp_raster - start_timestamp
        if trans_raster.shape[0] > 0:
            max = max if trans_raster[0].max() < max else trans_raster[0].max()
            min = min if trans_raster[0].min() > max else trans_raster[0].min()
            raster += trans_raster.tolist()

    n_bins = int(np.floor((max - min) / bin_width_s))
    plt.figure() #初始化一张图
    plt.hist(raster, n_bins)  #直方图关键操作
    plt.xlabel('time')
    plt.ylabel('trail') 
    plt.title(f'{direction}_psth') 
    plt.savefig(f"./{direction}_psth.jpg")

def pca_and_sort(filename, chan_idx):
    with h5py.File(filename, "r") as f:
        # Get channel spikes
        n_sorted_units = f["wf"].shape[0] - 1
        unsorted_spike = f[f["wf"][0, chan_idx]][()].T
        print(unsorted_spike.shape)
        
        # x_values = np.arange(0, 1.92, 0.04)
        # for num in range(len(unsorted_spike)):
        #     y_values = unsorted_spike[num]
        #     # 将点绘制在图上
        #     # plt.scatter(x_values, y_values)	
        #     # 将点连起来
        #     plt.plot(x_values, y_values)
        # plt.savefig(f"./{chan_idx}_spike.jpg")

        from sklearn.decomposition import PCA
        pca_sk = PCA(n_components=8)
        newMat = pca_sk.fit_transform(unsorted_spike)
        # plt.scatter(newMat[:, 0], newMat[:, 1])
        # plt.savefig(f"./{chan_idx}_pca.jpg")

        from sklearn.cluster import KMeans
        n_clusters = 3
        cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(newMat)
        # index = cluster.predict(newMat)
        # plt.scatter(newMat[:, 0], newMat[:, 1], c=index)
        # plt.savefig(f"./{chan_idx}_kmeans.jpg")

        gini_index = 0
        dataset_len = 0
        len = []
        gini_len = []
        for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
            spike_times = f[f["wf"][unit_idx, chan_idx]][()].T
            if spike_times.shape[0] < 2000:
                continue
            if spike_times.shape == (2,):
                # ignore this case (no data)
                continue
            dataset_len += spike_times.shape[0]
            len.append(spike_times.shape[0])
            temp_mat = pca_sk.transform(spike_times)
            temp = cluster.predict(temp_mat)
            nique_idxs, counts = np.unique(temp, return_counts=True)
            print(nique_idxs, counts)
            temp_p = counts / np.sum(counts)
            gini = 1 - np.sum(temp_p**2)
            gini_len.append(gini)
        pre = np.array(len) / dataset_len
        gini_index = pre * (np.array(gini_len).reshape(1,-1))
        print(np.sum(gini_index))

def get_vel_bins(filename, bin_width_s=.05, thresh=5000):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            min_t = t[0]
            binned_spikes = []
            num = 0
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times <= max_t]
                    spike_times = spike_times[spike_times >= min_t]
                    # make sure to ignore the hash here...
                    if len(spike_times) > thresh:
                        binned_spikes.append(spike_times)
        t_start=min_t
        t_end=max_t
        downsample_factor=1
        #Bin neural data using "bin_spikes" function
        neural_data=bin_spikes(binned_spikes, bin_width_s, t_start, t_end)
        print(neural_data.shape)
        # get vel 
        cursor_pos = f["cursor_pos"][:].T
        vel = np.diff(cursor_pos, axis=0, prepend=cursor_pos[0].reshape(1,-1)) / 0.004
        vel[0] = vel[1]
        acc = np.diff(vel, axis=0, prepend=vel[0].reshape(1,-1)) / 0.004
        acc[0] = acc[2]
        acc[1] = acc[2]

        output = np.concatenate((cursor_pos,vel, acc), axis=1)
        vels_binned=bin_output(output,t,bin_width_s,t_start,t_end,downsample_factor)
        print(vels_binned.shape)
    import pickle

    data_folder='data/' #FOLDER YOU WANT TO SAVE THE DATA TO

    with open(data_folder+'indy_0627.pickle','wb') as f:
        pickle.dump([neural_data,vels_binned],f)

    return result

def bin_spikes(spike_times,dt,wdw_start,wdw_end,):
    """
    Function that puts spikes into bins

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins

    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    num_neurons=len(spike_times) #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data

def bin_output(outputs,output_times,dt,wdw_start,wdw_end,downsample_factor=1):
    """
    Function that puts outputs into bins

    Parameters
    ----------
    outputs: matrix of size "number of times the output was recorded" x "number of features in the output"
        each entry in the matrix is the value of the output feature
    output_times: a vector of size "number of times the output was recorded"
        each entry has the time the output was recorded
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for binning the outputs
    wdw_end: number (any format)
        the end time for binning the outputs
    downsample_factor: integer, optional, default=1
        how much to downsample the outputs prior to binning
        larger values will increase speed, but decrease precision

    Returns
    -------
    outputs_binned: matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """

    ###Downsample output###
    #We just take 1 out of every "downsample_factor" values#
    if downsample_factor!=1: #Don't downsample if downsample_factor=1
        downsample_idxs=np.arange(0,output_times.shape[0],downsample_factor) #Get the idxs of values we are going to include after downsampling
        outputs=outputs[downsample_idxs,:] #Get the downsampled outputs
        output_times=output_times[downsample_idxs] #Get the downsampled output times

    ###Put outputs into bins###
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    output_dim=outputs.shape[1] #Number of output features
    outputs_binned=np.empty([num_bins,output_dim]) #Initialize matrix of binned outputs
    #Loop through bins, and get the mean outputs in those bins
    for i in range(num_bins): #Loop through bins
        idxs=np.where((np.squeeze(output_times)>=edges[i]) & (np.squeeze(output_times)<edges[i+1]))[0] #Indices to consider the output signal (when it's in the correct time range)
        for j in range(output_dim): #Loop through output features
            outputs_binned[i,j]=np.mean(outputs[idxs,j])

    return outputs_binned
###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.NaN
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X

def decode_linear_vel():
    data_folder='data/'
    with open(data_folder+'indy_0627.pickle','rb') as f:
        neural_data,vels_binned=pickle.load(f,encoding='latin1')
    bins_before=6 #How many bins of neural data prior to the output are used for decoding
    bins_current=1 #Whether to use concurrent time bin of neural data
    bins_after=6 #How many bins of neural data after the output are used for decoding
    # Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
    # Function to get the covariate matrix that includes spike history from previous bins
    X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)

    # Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
    #Put in "flat" format, so each "neuron / time" is a single feature
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    y=vels_binned
    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.8]
    testing_range=[0.8, 0.9]
    valid_range=[0.9,1]
    num_examples=X.shape[0]

    #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    training_set=np.arange(np.int32(np.round(training_range[0]*num_examples))+bins_before,np.int32(np.round(training_range[1]*num_examples))-bins_after)
    testing_set=np.arange(np.int32(np.round(testing_range[0]*num_examples))+bins_before,np.int32(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set=np.arange(np.int32(np.round(valid_range[0]*num_examples))+bins_before,np.int32(np.round(valid_range[1]*num_examples))-bins_after)

    #Get training data
    X_train=X[training_set,:,:]
    X_flat_train=X_flat[training_set,:]
    y_train=y[training_set,:]

    #Get testing data
    X_test=X[testing_set,:,:]
    X_flat_test=X_flat[testing_set,:]
    y_test=y[testing_set,:]

    #Get validation data
    X_valid=X[valid_set,:,:]
    X_flat_valid=X_flat[valid_set,:]
    y_valid=y[valid_set,:]   
    #Z-score "X" inputs. 
    X_train_mean=np.nanmean(X_train,axis=0)
    X_train_std=np.nanstd(X_train,axis=0)
    X_train=(X_train-X_train_mean)/X_train_std
    X_test=(X_test-X_train_mean)/X_train_std
    X_valid=(X_valid-X_train_mean)/X_train_std

    #Z-score "X_flat" inputs. 
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)
    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

    #Zero-center outputs
    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean
    y_valid=y_valid-y_train_mean 

    from sklearn import linear_model
    #Declare model
    model_wf=linear_model.LinearRegression()

    #Fit model
    model_wf.fit(X_flat_train,y_train)

    #Get predictions
    y_valid_predicted_wf=model_wf.predict(X_flat_valid)

    #Get metric of fit
    R2s_wf=get_R2(y_valid,y_valid_predicted_wf)
    print('R2s:', R2s_wf)

def decode_lstm_vel():
    data_folder='data/'
    with open(data_folder+'indy_0627.pickle','rb') as f:
        neural_data,vels_binned=pickle.load(f,encoding='latin1')
    bins_before=6 #How many bins of neural data prior to the output are used for decoding
    bins_current=1 #Whether to use concurrent time bin of neural data
    bins_after=6 #How many bins of neural data after the output are used for decoding
    # Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
    # Function to get the covariate matrix that includes spike history from previous bins
    X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)

    # Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
    #Put in "flat" format, so each "neuron / time" is a single feature
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    y=vels_binned
    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.8]
    testing_range=[0.8, 0.9]
    valid_range=[0.9,1]
    num_examples=X.shape[0]

    #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    training_set=np.arange(np.int32(np.round(training_range[0]*num_examples))+bins_before,np.int32(np.round(training_range[1]*num_examples))-bins_after)
    testing_set=np.arange(np.int32(np.round(testing_range[0]*num_examples))+bins_before,np.int32(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set=np.arange(np.int32(np.round(valid_range[0]*num_examples))+bins_before,np.int32(np.round(valid_range[1]*num_examples))-bins_after)

    #Get training data
    X_train=X[training_set,:,:]
    X_flat_train=X_flat[training_set,:]
    y_train=y[training_set,:]

    #Get testing data
    X_test=X[testing_set,:,:]
    X_flat_test=X_flat[testing_set,:]
    y_test=y[testing_set,:]

    #Get validation data
    X_valid=X[valid_set,:,:]
    X_flat_valid=X_flat[valid_set,:]
    y_valid=y[valid_set,:]   
    #Z-score "X" inputs. 
    X_train_mean=np.nanmean(X_train,axis=0)
    X_train_std=np.nanstd(X_train,axis=0)
    X_train=(X_train-X_train_mean)/X_train_std
    X_test=(X_test-X_train_mean)/X_train_std
    X_valid=(X_valid-X_train_mean)/X_train_std

    #Z-score "X_flat" inputs. 
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)
    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

    #Zero-center outputs
    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean
    y_valid=y_valid-y_train_mean 

    #Declare model
    model_lstm=LSTMRegression(units=512,dropout=0.1,num_epochs=10)

    #Fit model
    model_lstm.fit(X_train,y_train)

    #Get predictions
    y_valid_predicted_lstm=model_lstm.predict(X_valid)

    #Get metric of fit
    R2s_lstm=get_R2(y_valid,y_valid_predicted_lstm)
    print('R2s:', R2s_lstm)

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

def decode_kf_vel():
    data_folder='data/'
    with open(data_folder+'indy_0627.pickle','rb') as f:
        neural_data,vels_binned=pickle.load(f,encoding='latin1')
    
    lag=0 #What time bin of spikes should be used relative to the output
    X_kf=neural_data

    #For the Kalman filter, we use the position, velocity, and acceleration as outputs
    #The final output covariates include position, velocity, and acceleration
    y_kf=vels_binned[:, :4]
    num_examples=X_kf.shape[0]

    #Re-align data to take lag into account
    if lag<0:
        y_kf=y_kf[-lag:,:]
        X_kf=X_kf[0:num_examples+lag,:]
    if lag>0:
        y_kf=y_kf[0:num_examples-lag,:]
        X_kf=X_kf[lag:num_examples,:]
    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.8]
    testing_range=[0.8, 0.9]
    valid_range=[0.9,1]
    #Number of examples after taking into account bins removed for lag alignment
    num_examples_kf=X_kf.shape[0]
            
    #Note that each range has a buffer of 1 bin at the beginning and end
    #This makes it so that the different sets don't include overlapping data
    training_set=np.arange(np.int32(np.round(training_range[0]*num_examples_kf))+1,np.int32(np.round(training_range[1]*num_examples_kf))-1)
    testing_set=np.arange(np.int32(np.round(testing_range[0]*num_examples_kf))+1,np.int32(np.round(testing_range[1]*num_examples_kf))-1)
    valid_set=np.arange(np.int32(np.round(valid_range[0]*num_examples_kf))+1,np.int32(np.round(valid_range[1]*num_examples_kf))-1)

    #Get training data
    X_kf_train=X_kf[training_set,:]
    y_kf_train=y_kf[training_set,:]

    #Get testing data
    X_kf_test=X_kf[testing_set,:]
    y_kf_test=y_kf[testing_set,:]

    #Get validation data
    X_kf_valid=X_kf[valid_set,:]
    y_kf_valid=y_kf[valid_set,:]

    #Z-score inputs 
    X_kf_train_mean=np.nanmean(X_kf_train,axis=0)
    X_kf_train_std=np.nanstd(X_kf_train,axis=0)
    X_kf_train=(X_kf_train-X_kf_train_mean)/X_kf_train_std
    X_kf_test=(X_kf_test-X_kf_train_mean)/X_kf_train_std
    X_kf_valid=(X_kf_valid-X_kf_train_mean)/X_kf_train_std

    #Zero-center outputs
    y_kf_train_mean=np.mean(y_kf_train,axis=0)
    y_kf_train=y_kf_train-y_kf_train_mean
    y_kf_test=y_kf_test-y_kf_train_mean
    y_kf_valid=y_kf_valid-y_kf_train_mean

    #Declare model
    model_kf=KalmanFilterRegression(C=1) #There is one optional parameter that is set to the default in this example (see ReadMe)

    #Fit model
    model_kf.fit(X_kf_train,y_kf_train)
    print('start predict')
    #Get predictions
    y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid)

    #Get metrics of fit (see read me for more details on the differences between metrics)
    #First I'll get the R^2
    R2_kf=get_R2(y_kf_valid,y_valid_predicted_kf)
    print('R2:',R2_kf) #I'm just printing the R^2's of the 3rd and 4th entries that correspond to the velocities
    #Next I'll get the rho^2 (the pearson correlation squared)
    rho_kf=get_rho(y_kf_valid,y_valid_predicted_kf)
    print('rho2:',rho_kf**2) #I'm just printing the rho^2's of the 3rd and 4th entries that correspond to the velocities

########## Pearson's correlation (rho) ##########

def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """

    rho_list=[] #Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho) #Append rho of this output to the list
    rho_array=np.array(rho_list)
    return rho_array #Return the array of rhos

## Make trial data
if __name__ == '__main__':
    fname = 'data/indy_20160627_01.mat'
    data = load_data(fname)
    # 画神经元raster图
    # plot_raster(data, 'left')
    # 画神经元PSTH图
    # plot_psth(data, 'left')

    # 画神经元tuning curve图
    # unsorted = np.zeros(20, dtype=np.int32)
    # list = []
    # for num in range(124):
    #     r2 = plot_tuning_curve(data, num)
    #     id = int(r2 / 0.05)
    #     unsorted[id] += 1
    #     list.append(r2)
    # print(unsorted)
    # plt.figure() #初始化一张图
    # plt.hist(list, 20, range=(0,1))  #直方图关键操作
    # plt.xlabel('R2')
    # plt.ylabel('cell') 
    # plt.savefig(f"./check_R2.jpg")

    # 获取数据并计算速度与加速度
    # get_vel_bins(fname)
    
    # kf进行解码
    # decode_kf_vel()

    # 线性回归进行解码
    # decode_linear_vel()

    # lstm进行解码
    # decode_lstm_vel()

    # 神经元PCA降维和分类
    pca_and_sort(fname, 14)


    
