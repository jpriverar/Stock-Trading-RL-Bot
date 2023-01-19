import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import time

class StockTradingModel:
    def __init__(self, state_features, output_actions, samples):
        super().__init__()

        # Instanziating a scaler to transform the states before training
        self.scaler = StandardScaler()
        self.scaler.fit(samples)

        # Creating the multi output model
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=state_features, activation="relu"))
        self.model.add(Dense(len(output_actions), activation="linear"))
        self.model.compile(optimizer=SGD(learning_rate=0.001), loss="mse")

    def predict(self, state):
        # Simply scale the state and return all action-values for that state
        state = self.scaler.transform([state])
        end = time.time()
        return self.model.predict(state, verbose=0)

    def fit(self, state, action, target):
        # Scaling the state and getting the predictions for it
        scaled_state = self.scaler.transform([state])
        action_vals = self.predict(state)

        # Making the true results vector with the target and the predictions
        # from the current state
        expected_vals = np.array([target if a == action else val for a, val in enumerate(action_vals)])

        # Printing useful information
        self.model.fit(scaled_state, [expected_vals], verbose=0)

