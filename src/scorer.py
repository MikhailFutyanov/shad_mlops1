import pandas as pd
import logging
from lightgbm import LGBMClassifier
import lightgbm as lgb

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model

model = lgb.Booster(model_file='./models/lgb_model.bin')

# Define optimal threshold
model_th = 0.5
logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, path_to_file):

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model.predict(dt) > model_th).astype(int)
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission
