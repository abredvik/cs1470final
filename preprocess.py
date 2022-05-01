import numpy as np

def get_data(file_path):
    '''
    Method that retrieves data from a csv

    file_path: file path to the csv

    returns: a 2D numpy array of shape [len(data), 7]
    '''
    print('Retrieving data...')
    with open(file_path, 'r') as file:
        # ignore first two lines
        data = file.readlines()[2:]
        # initialize numerical data
        num_data = np.zeros([len(data), 7], dtype=float)
        for i, row in enumerate(data):
            row = row.replace('\n', '')
            row = np.array(row.split(','))
            # ignore non-numerical info
            num_data[i] = row[3:].astype(float)
    return num_data

def process(data, inter_length=100, overlap=90):
    '''
    Method to preprocess and batch data

    data: a 2D numpy array containing the data
    inter_length: length of intervals (in minutes)
    overlap: amount of overlap between intervals
    '''
    print('Processing data...')

    diff = inter_length - overlap

    # total number of inter_length intervals
    num_intervals = int((len(data) - inter_length) / diff)

    # shape [num_intervals, timesteps, features]
    batched_data = np.zeros([num_intervals, overlap, 7])
    labels = np.zeros([num_intervals,])

    for i in range(num_intervals):
        interval = data[i*diff:i*diff+inter_length]
        # min-max scaling
        mins = np.min(interval, axis=0)
        maxs = np.max(interval, axis=0)
        interval = (interval - mins) / (maxs - mins)
        # get rid of any NaNs
        interval[np.isnan(interval)] = 0
        # 90 minutes of previous data
        batched_data[i] = interval[:overlap]
        # is final close price greater than starting close price
        labels[i] = int(interval[-1, 3] > interval[-10, 3])

    print('Finished processing')
    return batched_data, labels