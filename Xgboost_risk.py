import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# data load
file = tf.keras.utils
raw_df_base = pd.read_csv('C:/Users/user/Desktop/TIMC/송수관,배수관 분석/Classifier/3년치 데이터/학습용/data_xg_ds_before_3y.csv')

# data preprocessing
onehot_cols = ['SAA_CDE', 'MOP_CD', 'DRNG_CD', 'DBTG_CD', 'USEG_CD', 'ROAD_CD']
raw_df = pd.get_dummies(raw_df_base, columns=onehot_cols)

normal, nusu = np.bincount(raw_df['LEAK_CHK'])
total = normal + nusu

cleaned_df = raw_df.copy()
cleaned_df = cleaned_df.drop(columns=["GISID", "BUILD_Y", "year", "DOT_FREQ"])

train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=110)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=110)


train_labels = np.array(train_df.pop('LEAK_CHK'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('LEAK_CHK'))
test_labels = np.array(test_df.pop('LEAK_CHK'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

nusu_df = pd.DataFrame(train_features[bool_train_labels], columns=train_df.columns)
normal_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

# log
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(
            32, activation='relu',
            input_shape=(train_features.shape[-1],)),
        keras.layers.Dense(
            32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(
            16, activation='relu'),
        keras.layers.Dense(
            16, activation='relu'),
        keras.layers.Dense(
            8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

EPOCHS = 500
BATCH_SIZE = 1024
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


initial_bias = np.log([nusu/normal])


model = make_model(output_bias=initial_bias)
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)


model_new = make_model()
model_new.load_weights(initial_weights)
baseline_history = model_new.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels))

def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.5, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

plot_metrics(baseline_history)

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    print('Total f1score: ', (2 * (cm[1][1]/(cm[1][1]+cm[0][1]))*(cm[1][1]/(cm[1][1]+cm[1][0])))/((cm[1][1]/(cm[1][1]+cm[0][1]))+(cm[1][1]/(cm[1][1]+cm[1][0]))))
    print('Recall: ', (cm[1][1]/(cm[1][1]+cm[1][0])))
    print('Precition: ', (cm[1][1]/(cm[1][1]+cm[0][1])))
    print()
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

test_predictions_weighted = model_new.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = model_new.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model_new.metrics_names, weighted_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)

#위험도 데이터셋 load
raw_df_base_Risk = pd.read_csv('C:/Users/user/Desktop/TIMC/송수관,배수관 분석/Classifier/3년치 데이터/위험도용/data_xg_ds_now_3y.csv')

onehot_cols_Risk = ['SAA_CDE', 'MOP_CD', 'DRNG_CD', 'DBTG_CD', 'USEG_CD', 'ROAD_CD']
raw_df_Risk = pd.get_dummies(raw_df_base_Risk, columns=onehot_cols_Risk)

cleaned_df_Risk = raw_df_Risk.copy()
cleaned_df_Risk = cleaned_df_Risk.drop(columns=["GISID", "BUILD_Y", "year"])
test_labels_Risk = np.array(cleaned_df_Risk.pop('LEAK_CHK'))
test_features_Risk = np.array(cleaned_df_Risk)

scaler = MinMaxScaler()

test_features_Rist_min = scaler.fit_transform(test_features_Risk)
remain_Risk_df = model_new.predict(test_features_Rist_min, batch_size=BATCH_SIZE)
remain_Risk = raw_df_Risk.copy()
remain_Risk['remain_life'] = remain_Risk_df

#결과값 저장
remain_Risk.to_csv('C:/Users/user/Desktop/TIMC/송수관,배수관 분석/1차안/3년치_S/DNN_predict_final.csv', index=False, encoding='cp949')