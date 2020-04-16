import timeit
start = timeit.default_timer()
import pandas as pd
import numpy as np
import warnings
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
from keras.utils import model_to_dot


lab = [ 0,  4,  1,  9,  6, 15, 11,  2, 10,  8,  3, 14,  7, 13,  5, 12]
cols = ['Education', 'Lodging/residential', 'Office',
       'Entertainment/public assembly', 'Other', 'Retail', 'Parking',
       'Public services', 'Warehouse/storage', 'Food sales and service',
       'Religious worship', 'Healthcare', 'Utility', 'Technology/science',
       'Manufacturing/industrial', 'Services']
label = {cols[i]:lab[i] for i in range(len(cols))}
def getSavedModel(kind):
    with open('modelsList.pkl', 'rb') as f:
        mdls = pickle.load(f)
    with open('historyList.pkl', 'rb') as f:
        hst = pickle.load(f)
    return mdls,hst

categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday",  "meter"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale',"sea_level_pressure", "wind_direction"]

feat_cols = categoricals + numericals
def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols
    X = {col: np.array(df[col]) for col in cols}
    return X
def run(ip):
    inpCol = ['row_id', 'building_id', 'meter', 'timestamp', 'site_id', 'primary_use', 'square_feet', 'year_built', 'floor_count', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    sample = pd.DataFrame(np.array(ip).reshape(1,16),columns = inpCol)
    for i in sample.columns:
        if i == 'timestamp':
            continue
        sample[i] = float(sample[i])
    sample["timestamp"] = pd.to_datetime(sample["timestamp"])
    sample["hour"] = sample["timestamp"].dt.hour
    sample["weekday"] = sample["timestamp"].dt.weekday

    sample['wind_speed'].fillna((sample['wind_speed'].mean()), inplace=True)

    classifyWind = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 
            (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

    for item in classifyWind:
        sample.loc[(sample['wind_speed']>=item[1]) & (sample['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

    res = np.zeros((sample.shape[0]),dtype=np.float32)
    fold = 2
    for_prediction = get_keras_data(sample, numericals, categoricals)
    res[0] = np.expm1(sum([model.predict(for_prediction, batch_size=1) for model in models['model'+str(int(for_prediction['meter'][0]))]])/fold)
    print('Predicted Meter reading for given inputs : ',res[0],' units')
models,history = getSavedModel('Imputed')
ip = ['107','107','0','2016-01-01 00:00:00','3',label['Education'],'97532','2005','10','3.8','255','2.4','1','1020.9','240','3.1']
run(ip)
print(timeit.default_timer()-start)