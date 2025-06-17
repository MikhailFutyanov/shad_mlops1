# Import standard libraries
import pandas as pd
import numpy as np
import logging

# Import extra modules
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer 
from category_encoders import CatBoostEncoder

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

def add_time_features(df):
    logger.debug('Adding time features...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df.drop(columns='transaction_time', inplace=True)
    return df

def add_ratio_feature(df):
    logger.debug('Adding ratio feature...')
    # Вычисляем среднюю сумму транзакций по продавцу (merch)
    df['merchant_mean_amount'] = df.groupby('merch')['amount'].transform('mean')
    # Создаем новый признак: отношение суммы к средней сумме по продавцу
    df['amount_to_mean_ratio'] = df['amount'] / df['merchant_mean_amount']
    # Удаляем временный признак merchant_mean_amount
    df.drop(columns='merchant_mean_amount', inplace=True)
    return df



def add_distance_features(df):
    
    logger.debug('Calculating distances...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']), 
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])

def drop_unnecessary_columns(df):
    logger.debug('Dropping unnecessary columns...')
    return df.drop(columns=['gender', 'one_city', 'us_state', 'jobs', 'merch'])


# Calculate means for encoding at docker container start
def load_train_data():

    logger.info('Loading training data...')

    # Define column types
    target_col = 'target'
    categorical_cols = ['cat_id']

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv').drop(columns=['name_1', 'name_2', 'street', 'post_code'])
    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some simple time features
    train = add_time_features(train)

    encoder = CatBoostEncoder(cols=categorical_cols, 
                              return_df=True,
                              random_state=42)
    train[categorical_cols] = encoder.fit_transform(train[categorical_cols], train[target_col])
    logger.info('CatBoost кодирование применено к столбцам: %s', categorical_cols)
    
    # Calculate distance between a client and a merchant
    train = add_distance_features(train)


    train = add_ratio_feature(train)
    train = drop_unnecessary_columns(train)
    
    logger.info('Train data processed. Shape: %s', train.shape)

    return train, encoder


# Main preprocessing function
def run_preproc(train, input_df, encoder):

    # Define column types
    target_col = 'target'
    categorical_cols = ['cat_id']

    continuous_cols = ['amount', 'population_city']
    
    # Run category encoding
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    logger.info('Categorical encoding completed: %s', input_df.shape)
    
    # Add some simple time features
    input_df = add_time_features(input_df)
    logger.info('Added time features. Output shape: %s', input_df.shape)

    categorical_cols.extend(['hour'])
    
    # Заполнение пропусков в категориальных признаках модой
    for col in categorical_cols:
        mode_val = train[col].mode()[0]
        input_df[col].fillna(mode_val, inplace=True)

    logger.info('Заполнение пропусков в категориальных признаках завершено...')

    # Calculate distance between a client and a merchant
    input_df = add_distance_features(input_df)
    continuous_cols.extend(['distance'])


    # Добавление отношения суммы транзакции к средней сумме транзакций по продавцу
    input_df = add_ratio_feature(input_df)
    continuous_cols.extend(['amount_to_mean_ratio'])

    # Заполнение пропусков в непрерывных признаках медианой
    imputer = SimpleImputer(missing_values=np.nan, strategy='median') 
    imputer = imputer.fit(train[continuous_cols])
    input_df[continuous_cols] = imputer.transform(input_df[continuous_cols])
    
    input_df = drop_unnecessary_columns(input_df)

    
    logger.info('Continuous features preprocessing completed. Output shape: %s', input_df.shape)
    
    # Return resulting dataset
    return input_df