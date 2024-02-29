from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from joblib import parallel_backend, dump, load

class neural:
    def run():
        x, y = neural.get_ready()

        #try:
        #    model = load('neuro1.joblib')
        #except:
        model = neural.training(x, y)

        y_ = neural.predict(model, x)
        neural.accuracy_test(y, y_)

        return None

    def get_ready():
        raw_x = pd.read_csv("base.csv")
        x = raw_x.iloc[:-round(len(raw_x)*0.8)]

        raw_y = raw_x.iloc[-round(len(raw_x)*0.2):]
        y = raw_y["Close"]

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        return x, y

    def training(x, y):
        with parallel_backend('loky', n_jobs=8):
            model = MLPRegressor(hidden_layer_sizes= 4096,
                                    learning_rate_init=0.07 ,
                                    max_iter=4000,
                                    activation="tanh",
                                    learning_rate="adaptive")

            epochs = model.max_iter
            train_bar = tqdm(total=epochs, desc="Fitting")

            for _ in range(epochs):
                model.partial_fit(x, y)
                train_bar.update(1)

            train_bar.close()

            dump(model, 'neuro_4K_4096_007_TAN_ADAP.joblib')

            return model

    def predict(model, x):
        return model.predict(x)

    def accuracy_test(y, y_):
        accuracy_bar = tqdm(total=5, desc="Accuracy test")

        R2 = r2_score(y, y_)
        accuracy_bar.update(1)
        MAE = mean_absolute_error(y, y_)
        accuracy_bar.update(1)
        MAPE = mean_absolute_percentage_error(y, y_)
        accuracy_bar.update(1)
        MSE = ((y - y_) ** 2).mean()
        accuracy_bar.update(1)

        date_log = pd.read_csv("dates.csv")

        #y = pd.DataFrame(y)
        #y_ = pd.DataFrame(y_)

        #y.to_csv("y.csv")
        #y_.to_csv("y_.csv")

        plt.figure(figsize=(12, 6))

        plt.plot(date_log["Date"].iloc[-int(len(date_log) * 0.2):], y, label = 'Close')
        plt.pause(1)
        plt.plot(date_log["Date"].iloc[-int(len(date_log) * 0.2):], y_, label = 'Close_predict')
        plt.pause(1)

        plt.legend()

        accuracy_bar.update(1)

        accuracy_bar.close()

        results_table = f"""
                        RESULTS:
                        R2 = {R2}
                        MAE = {MAE}
                        MAPE = {MAPE}
                        MSE = {MSE}"""

        print(results_table)
        plt.show()
        return None

network = neural.run()
