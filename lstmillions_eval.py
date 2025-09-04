# Students
# Chandler Burgess, clb015210
# Wai Yan Phyoe, wxp220006

import enum
import itertools
import logging
import os
import sys
from collections import defaultdict

# This flag is used to suppress the TensorFlow info and warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from timeit import default_timer as timer
from typing import Optional

import keras
import keras.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TENSORFLOW_VERBOSITY = 0


class Indicators(enum.Enum):
    RSI_14 = 1
    EMA_20 = 2
    EMA_50 = 3
    OBV = 4


class Target(enum.Enum):
    PCT_CHANGE = 2
    UP_DOWN = 3


class StockDataset:
    def __init__(
        self,
        ticker: str,
        target: Target = Target.PCT_CHANGE,
        indicators: Optional[Indicators] = None,
        drop_features: list = None,
    ):
        self.ticker = ticker
        self.indicators = indicators

        logging.info(f"Loading data for {ticker}....")
        data = self._download(ticker)
        logging.info("Enhancing with tehcnical indicators....")
        if indicators and isinstance(indicators, list):
            self._enhance(data, indicators)

        logging.info("Preparing data....")
        data.drop("Close", axis=1, inplace=True)
        if drop_features and isinstance(drop_features, list):
            if "Adj Close" in drop_features:
                raise ValueError("Cannot drop 'Adj Close' feature")

            for feature in drop_features:
                if feature not in data.columns:
                    raise ValueError(f"Feature '{feature}' not in data")

            data.drop(drop_features, axis=1, inplace=True)

        self.features = data.columns.tolist()
        self._add_targets(data)
        data.dropna(inplace=True)

        self.data = data
        # log the number of samples and features
        logging.info(f"Samples: {data.shape[0]}, Features: {len(self.features)}")
        logging.info(f"DONE loading data for {ticker}\n")

    def _download(self, ticker):
        if not os.path.exists(f"{ticker}.csv"):
            data = yf.download(ticker)
            data.to_csv(f"{ticker}.csv")
        else:
            data = pd.read_csv(f"{ticker}.csv", index_col=0, parse_dates=True)

        return data

    def _enhance(self, data, indicators):
        if Indicators.RSI_14 in indicators:
            data["RSI"] = ta.momentum.rsi(data["Adj Close"], window=14)

        if Indicators.EMA_20 in indicators:
            data["EMA_20"] = ta.trend.ema_indicator(data["Adj Close"], window=20)

        if Indicators.EMA_50 in indicators:
            data["EMA_50"] = ta.trend.ema_indicator(data["Adj Close"], window=50)

        if Indicators.OBV in indicators:
            data["OBV"] = ta.volume.on_balance_volume(data["Adj Close"], data["Volume"])

    def _add_targets(self, data):
        # name of the target column is enum name
        data[Target.PCT_CHANGE.name] = data["Adj Close"].pct_change()
        data[Target.UP_DOWN.name] = np.where(
            data["Adj Close"] > data["Adj Close"].shift(1), 1, 0
        )

        # Remove first row since in either case it will be junk since we're comparing
        # each row to the previous row and their isnt a previous row for the first row
        data = data[1:]

    def train_test_split(self, target: Target, seq_length: int, test_period: int):
        assert seq_length > 0, "Sequence length must be greater than 0"
        assert test_period > 0, "Test window size must be greater than 0"

        train_data = self.data[:-test_period]
        test_data = self.data[-(test_period + seq_length) :]

        X_train = train_data[self.features]
        y_train = train_data[[target.name]]
        X_test = test_data[self.features]
        y_test = test_data[[target.name]][seq_length:]

        return X_train, X_test, y_train, y_test


class LSTMModel:
    def __init__(
        self,
        target: Target,
        hidden_units: int,
        num_lstm_layers: int = 1,
        seq_length: int = 14,
        epochs: int = 10,
        batch_size: int = 32,
    ):
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size

        logging.info(
            f"Initializing LSTM model with {hidden_units} hidden units and {num_lstm_layers} LSTM layers"
        )
        self.model = Sequential()

        for _ in range(num_lstm_layers - 1):
            self.model.add(LSTM(units=hidden_units, return_sequences=True))
        self.model.add(LSTM(units=hidden_units))

        if target == Target.PCT_CHANGE:
            # This is a regression problem so we use a linear activation function
            # because we want to predict a continuous value
            self.model.add(Dense(1))
            # We use the mean squared error loss function because this is a regression problem
            self.model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[
                    keras.metrics.RootMeanSquaredError(),
                    keras.metrics.MeanSquaredError(),
                ],
            )
        elif target == Target.UP_DOWN:
            # This is a classification problem so we use a sigmoid activation function
            # because we want to predict a binary value
            self.model.add(Dense(1, activation="sigmoid"))
            # We use the binary crossentropy loss function because this is a classification problem
            self.model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

        logging.info("DONE initializing LSTM model\n")

    def _create_training_sequences(self, X, y):
        xs, ys = [], []
        for i in range(X.shape[0] - self.seq_length):
            xseq = X[i : i + self.seq_length]
            yseq = y[i + self.seq_length]
            xs.append(xseq)
            ys.append(yseq)

        return np.array(xs), np.array(ys)

    def fit(self, X, y):
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        logging.info("Preparing training sequences....")
        X_train, y_train = self._create_training_sequences(
            self.X_scaler.fit_transform(X), self.y_scaler.fit_transform(y)
        )

        # shuffle the X_train and y_train since
        # validation_split will take the last 10% of the training
        # and we want to make sure it's not just the last 10% of the data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        logging.info("Fitting model....")
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=TENSORFLOW_VERBOSITY,
            shuffle=False,  # we already shuffled the data
            validation_split=0.1,
        )
        logging.info("DONE fitting model\n")
        return history

    def _create_sequences(self, X):
        xs = []
        for i in range(X.shape[0] - self.seq_length):
            x = X[i : i + self.seq_length]
            xs.append(x)

        return np.array(xs)

    def predict(self, X):
        X_test = self._create_sequences(self.X_scaler.transform(X))
        y_pred = self.model.predict(X_test, verbose=TENSORFLOW_VERBOSITY)
        return self.y_scaler.inverse_transform(y_pred)


def plot_pct_validation_curve(history, run_params):
    plt.figure(figsize=(15, 8.5))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="loss train")
    plt.plot(history.history["val_loss"], label="loss validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(f"{ticker} Loss Curve", loc="right")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["root_mean_squared_error"], label="rmse train")
    plt.plot(history.history["val_root_mean_squared_error"], label="rmse validation")
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")
    plt.legend(loc="upper right")
    plt.title(f"{ticker} RMSE Curve", loc="right")

    plt.suptitle(f"Validation for{run_params}")
    plt.savefig(f"{ticker}_PCT_validation_curve.png")
    plt.show()
    plt.close()


def plot_ud_validation_curve(history, run_params, predictions, probs, actuals):
    # PLot confusion matrix using scikit-learn ConfustionMatrixDisplay
    disp = ConfusionMatrixDisplay.from_predictions(
        actuals, predictions, cmap=plt.cm.Blues
    )
    disp.ax_.set_title(f"{ticker.upper()} Confusion Matrix")
    plt.savefig(f"{ticker}_UD_confusion_matrix.png")
    plt.show()
    plt.close()

    false_pos_rate, true_pos_rate, _ = roc_curve(actuals, probs)
    auc_score = auc(false_pos_rate, true_pos_rate)

    plt.plot([0, 1], [0, 1], "x--")
    plt.plot(false_pos_rate, true_pos_rate, label="AUC = {:.3f})".format(auc_score))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{ticker} ROC curve")
    plt.savefig(f"{ticker}_UD_roc_curve.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(11, 8.5))
    plt.subplots_adjust(wspace=0.25)

    plt.subplot(1, 2, 1)
    loss_curve = history.history["loss"]
    val_loss_curve = history.history["val_loss"]
    plt.plot(loss_curve, label="loss train")
    plt.plot(val_loss_curve, label="loss validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(f"{ticker} Loss Curve", loc="right")

    plt.subplot(1, 2, 2)
    accuracy_curve = history.history["accuracy"]
    val_accuracy_curve = history.history["val_accuracy"]
    plt.plot(accuracy_curve, label="accuracy train")
    plt.plot(val_accuracy_curve, label="accuracy validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.title(f"{ticker} Accuracy Curve", loc="right")

    plt.suptitle(f"Validation for{run_params}")
    plt.savefig(f"{ticker}_UD_validation_curve.png")
    plt.show()
    plt.close()


def plot_accuracy_curve(historys, labels, title):
    plt.figure(figsize=(11, 8.5))
    for history, label in zip(historys, labels):
        accuracy_curve = history.history["accuracy"]
        plt.plot(accuracy_curve, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{ticker} Accuracy Curve {title}", loc="right")
    plt.legend(
        labels,
        loc=3,
        bbox_to_anchor=(0, 1.05, 1, 0.102),
        mode="expand",
        ncol=7,
        borderaxespad=0,
        title="seq_length, hidden_units, num_layers, epochs",
    )
    plt.savefig(f"{ticker}_UD_accuracy_curve_{title}.png")
    plt.show()
    plt.close()


def plot_loss_curve(historys, labels, title):
    plt.figure(figsize=(11, 8.5))
    for history, label in zip(historys, labels):
        loss_curve = history.history["loss"]
        plt.plot(loss_curve, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ticker} Loss Curve {title}", loc="right")

    plt.legend(
        labels,
        loc=3,
        bbox_to_anchor=(0, 1.05, 1, 0.102),
        mode="expand",
        ncol=7,
        borderaxespad=0,
        title="seq_length, hidden_units, num_layers, epochs, mse",
    )
    plt.savefig(f"{ticker}_PCT_loss_curve_{title}.png")
    plt.show()
    plt.close()


def model_ticker(
    ticker: str,
    test_period: int = 60,
    sequence_lengths: list[int] = [7],
    hidden_units: list[int] = [32],
    lstm_layers: list[int] = [1],
    epochs: list[int] = [10],
    targets: list[Target] = [Target.UP_DOWN],
):
    indicators = [
        Indicators.RSI_14,
        Indicators.EMA_20,
        Indicators.EMA_50,
        Indicators.OBV,
    ]
    dataset = StockDataset(ticker, indicators=indicators)
    size_of_data = dataset.data.shape[0]

    run_params = defaultdict(list)
    historys = defaultdict(list)
    labels = defaultdict(list)
    for model_epochs, seq_length, hidden_units, num_layers, target in itertools.product(
        epochs, sequence_lengths, hidden_units, lstm_layers, targets
    ):

        params = {
            "seq_length": seq_length,
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            "epochs": model_epochs,
            "target": target.name,
        }
        logging.info("Running model: %s", params)

        X_train, X_test, y_train, y_test = dataset.train_test_split(
            target, seq_length, test_period
        )

        model = LSTMModel(
            target,
            hidden_units,
            num_lstm_layers=num_layers,
            seq_length=seq_length,
            epochs=model_epochs,
        )

        accuracy = None
        mse = None
        if target == Target.PCT_CHANGE:
            history = model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            actuals = y_test.values
            mse = mean_squared_error(actuals.flatten(), predictions.flatten())
            loss = history.history["loss"][-1]
            val_loss = history.history["val_loss"][-1]
            model_mse = history.history["mean_squared_error"][-1]
            val_mse = history.history["val_mean_squared_error"][-1]
            logging.info(f"Mean Squared Error: {mse}\n")
            logging.info(f"Model Loss: {loss}\n")
            logging.info(f"Validation Loss: {val_loss}\n")
            params["size_of_dataset"] = size_of_data
            params["testing_mse"] = mse
            params["training_loss"] = loss
            params["validation_loss"] = val_loss
            params["training_mse"] = model_mse
            params["validation_mse"] = val_mse
            params["history"] = history
            params["actuals"] = actuals
            params["predictions"] = predictions
            historys[target].append(history)
        elif target == Target.UP_DOWN:
            history = model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            actuals = y_test.values
            # Use the accuracy as the evaluation metric
            accuracy = (np.round(predictions) == actuals).mean()
            model_accuracy = history.history["accuracy"][-1]
            val_accuracy = history.history["val_accuracy"][-1]
            logging.info(f"Accuracy: {accuracy}\n")
            logging.info(f"Model Accuracy: {model_accuracy}\n")
            logging.info(f"Validation Accuracy: {val_accuracy}\n")
            params["size_of_dataset"] = size_of_data
            params["testing_accuracy"] = accuracy
            params["training_accuracy"] = model_accuracy
            params["validation_accuracy"] = val_accuracy
            params["history"] = history
            params["predictions"] = predictions
            params["y_test"] = y_test
            historys[target].append(history)

        run_params[target].append(params)

        ### All the evaluation code should go here
        labels[target].append(
            f"{seq_length}, {hidden_units}, {num_layers}, {model_epochs}"
        )

    end = timer()
    print(end - start)

    split_num = len(epochs)

    for target in targets:
        if target == Target.PCT_CHANGE:
            historys_parts = np.array_split(historys[target], split_num)
            labels_parts = np.array_split(labels[target], split_num)

            for i in range(split_num):
                plot_loss_curve(historys_parts[i], labels_parts[i], f"{i + 1}")

            target_params = pd.DataFrame(run_params[target])
            target_params = target_params.sort_values(
                by=["training_loss"], ascending=True
            )
            top_actuals = target_params.iloc[0]["actuals"]
            top_predictions = target_params.iloc[0]["predictions"]
            top_history = target_params.iloc[0]["history"]
            target_params = target_params.drop(
                columns=["history", "actuals", "predictions"]
            )

            # Plot actual vs predicted
            plt.figure(figsize=(11, 8.5))
            plt.plot(top_actuals, label="Actual")
            plt.plot(top_predictions, label="Predicted")
            plt.xlabel("Day")
            plt.ylabel("% Change")
            plt.legend(loc="upper right")
            plt.title(f"{ticker} Actual vs Predicted % Change")
            plt.savefig(f"{ticker}_PCT_actual_vs_predicted.png")
            plt.show()
            plt.close()

            top_params = (
                target_params.drop(
                    columns=[
                        "training_loss",
                        "validation_loss",
                        "training_mse",
                        "validation_mse",
                        "testing_mse",
                    ]
                )
                .iloc[0]
                .to_numpy()
            )

            plot_pct_validation_curve(top_history, top_params)

            target_params.to_csv(f"{ticker}_PCT_runs.csv", index=False)

        if target == Target.UP_DOWN:
            historys_parts = np.array_split(historys[target], split_num)
            labels_parts = np.array_split(labels[target], split_num)

            for i in range(split_num):
                plot_accuracy_curve(historys_parts[i], labels_parts[i], f"{i + 1}")

            target_params = pd.DataFrame(run_params[target])
            target_params = target_params.sort_values(
                by=["training_accuracy"], ascending=False
            )
            top_history = target_params.iloc[0]["history"]
            probs = target_params.iloc[0]["predictions"].ravel()

            # Plot the probability of up/down for each day
            plt.figure(figsize=(11, 8.5))
            plt.ylim(0, 1)
            plt.plot(probs, label="Probability")
            plt.xlabel("Day")
            plt.ylabel("Probability")
            plt.legend(loc="upper right")
            plt.title(f"{ticker} Probability of Up/Down")
            plt.savefig(f"{ticker}_UD_probability.png")
            plt.show()
            plt.close()

            predictions = np.round(probs)
            y_test = target_params.iloc[0]["y_test"]

            target_params = target_params.drop(
                columns=["history", "predictions", "y_test"]
            )
            top_params = (
                target_params.drop(
                    columns=[
                        "training_accuracy",
                        "validation_accuracy",
                        "testing_accuracy",
                    ]
                )
                .iloc[0]
                .to_numpy()
            )

            plot_ud_validation_curve(
                top_history, top_params, predictions, probs, y_test
            )

            target_params.to_csv(f"{ticker}_UD_runs.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python lstmillions.py <ticker>")
        sys.exit(1)

    ticker = sys.argv[1]
    start = timer()

    test_period = 60
    sequence_lengths = [7, 14, 28]
    hidden_units = [32, 64, 128]
    lstm_layers = [1, 2, 3]
    epochs = [10, 20]
    targets = [Target.UP_DOWN, Target.PCT_CHANGE]

    # sequence_lengths = [7, 14]
    # hidden_units = [32]
    # lstm_layers = [1]
    # epochs = [10]
    # targets = [Target.UP_DOWN, Target.PCT_CHANGE]

    model_ticker(
        ticker,
        test_period,
        sequence_lengths,
        hidden_units,
        lstm_layers,
        epochs,
        targets,
    )
