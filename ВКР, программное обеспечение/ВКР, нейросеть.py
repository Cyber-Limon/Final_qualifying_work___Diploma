import random
import joblib
import numpy  as np
import pandas as pd

from tensorflow                  import keras
from sklearn.preprocessing       import StandardScaler
from sklearn.model_selection     import train_test_split
from tensorflow.keras.layers     import Dense, Dropout
from tensorflow.keras.callbacks  import EarlyStopping
from tensorflow.keras.optimizers import Adam



def augmentation(X, y, quantity, bias):
    X_current = []
    y_current = []

    for material in range(len(X)):
        if y[material][material_property] != 0:
            X_current.append(X[material])
            y_current.append(y[material][material_property])

    X_aug = list(X_current.copy())
    y_aug = list(y_current.copy())

    while len(X_aug) < quantity:
        material  = random.randint(0, len(X) - 1)
        x_current = X[material].copy()

        sample = []
        for i in range(len(x_current)):
            if x_current[i] > bias:
                sample.append(i)

        if len(sample) >= 6:
            number_of_changes = 6
        elif len(sample) >= 4:
            number_of_changes = 4
        elif len(sample) >= 2:
            number_of_changes = 2
        else:
            continue

        changes = random.sample(sample, number_of_changes)
        for change in range(len(changes)):
            if change % 2 == 0:
                x_current[changes[change]] += bias
            else:
                x_current[changes[change]] -= bias

        X_aug.append(x_current)
        y_aug.append(y_current[material])

    return X_aug, y_aug



compound   = pd.read_excel("ВКР, датасет.xlsx", usecols="B:V")
compound   = compound.values.tolist()
compound   = np.array(compound)

properties = pd.read_excel("ВКР, датасет.xlsx", usecols="W:AD")
properties = properties.values.tolist()
properties = np.array(properties)



X_train, X_test, y_train, y_test = train_test_split(
    compound, properties,
    test_size   =0.2,
    random_state=1107,
    shuffle     =True)



scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(X_train)
X_test   = scaler_X.transform(X_test)

scaler_Y = StandardScaler()
y_train  = scaler_Y.fit_transform(y_train)
y_test   = scaler_Y.transform(y_test)

joblib.dump(scaler_X, "ВКР, нормализаторы\\scaler_X.pkl")
joblib.dump(scaler_Y, "ВКР, нормализаторы\\scaler_Y.pkl")



for material_property in range(len(properties[0])):
    X_train_current, y_train_current = augmentation(X_train, y_train, 2000, 0.01)
    X_test_current,  y_test_current  = augmentation(X_test,  y_test,   500, 0.01)

    X_train_current = np.array(X_train_current)
    X_test_current  = np.array(X_test_current)
    y_train_current = np.array(y_train_current)
    y_test_current  = np.array(y_test_current)



    model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),

        Dense(256, activation='relu'),
        Dropout(0.2),

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(64,  activation='relu'),
        Dropout(0.4),

        Dense(32,  activation='relu'),
        Dropout(0.5),

        Dense(1)])

    model.summary()



    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss     ='mse',
                  metrics  =['mae'])

    early_stopping = EarlyStopping(
        monitor             ='val_loss',
        patience            =20,
        restore_best_weights=True)

    log = model.fit(
        X_train_current, y_train_current,
        batch_size      =32,
        epochs          =100,
        validation_split=0.2,
        callbacks       =[early_stopping],
        verbose         =1)



    model.evaluate(X_test_current, y_test_current, verbose=1)



    model.save(f"ВКР, модели\\ВКР, модель {material_property + 1}.keras")
