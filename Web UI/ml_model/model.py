import numpy as np
import json

class L_layer_model:
    def __init__(self, layer_dims=None):
        self.layer_dims = layer_dims
        self.params = {}
    def relu(self, Z):
        cache = Z
        A = np.maximum(0, Z)
        return A, cache

    def softmax(self, Z):
        cache = Z
        exp_Z = np.exp(Z)
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return A, cache

    def init_params(self, layer_dims):
        L = len(layer_dims)
        params = {}
        for l in range(1, L):
            params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
            params[f'b{l}'] = np.zeros((layer_dims[l], 1))
        return params

    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def linear_forward(self, A, W, b):
        Z = W.dot(A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, f):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A, activation_cache = f(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = - (1./m) * np.sum(Y * np.log(AL))
        cost = np.squeeze(cost)
        return cost

    def linear_back(self, dZ, cache):
        A_prev, W, b = cache
        m = dZ.shape[1]
        dW = (1./m) * np.dot(dZ, A_prev.T)
        db = (1./m) * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def linear_activation_back(self, dA, cache, f):
        linear_cache, activation_cache = cache
        dZ = f(dA, activation_cache)
        dA_prev, dW, db = self.linear_back(self, dZ, linear_cache)
        return dA_prev, dW, db

    def update_params(self, params, grads, learning_rate):
        L = len(params) // 2
        for l in range(L):
            params[f'W{l + 1}'] = params[f'W{l + 1}'] - learning_rate * grads[f'dW{l + 1}']
            params[f'b{l + 1}'] = params[f'b{l + 1}'] - learning_rate * grads[f'db{l + 1}']
        return params

    def L_model_forward(self, X, params):
        caches = []
        A = X
        L = len(params) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, params[f'W{l}'], params[f'b{l}'], self.relu)
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, params[f'W{L}'], params[f'b{L}'], self.softmax)
        caches.append(cache)

        return AL, caches

    def L_model_backward(self, AL, Y, caches):
        L = len(caches)
        grads = {}
        current_cache = caches[L - 1]
        dAL = AL - Y
        grads[f'dA{L - 1}'], grads[f'dW{L}'], grads[f'db{L}'] = self.linear_back(dAL, current_cache[0])

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_back(grads[f"dA{l + 1}"], current_cache, self.relu_backward)
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l + 1}"] = dW_temp
            grads[f"db{l + 1}"] = db_temp

        return grads

    def L_layer_model(self, X, Y, layer_dims, learning_rate=.063, num_iter=3200, print_cost=False):
        costs = []
        L = len(layer_dims)
        params = self.init_params(layer_dims)
        for i in range(0, num_iter):
            AL, caches = self.L_model_forward(X, params)
            cost = self.compute_cost(AL, Y)
            costs.append(cost)
            if i % 100 == 0 and i != 0:
                print(f'Cost after iteration {i}: {cost}')
            grads = self.L_model_backward(AL, Y, caches)
            params = self.update_params(params, grads, learning_rate)
        
        return params, costs

    def predict(self, X, Y):
        m = X.shape[1]
        probas, caches = self.L_model_forward(X, self.params)
        p = np.argmax(probas, axis = 0)
        if Y is not None:
            y = np.argmax(Y, axis = 0)
            accuracy = 1./m * np.sum(p == y)
            print(f"Accuracy: {accuracy}")
        return p

    def one_hot(self, Y):
        m = len(Y)
        one_hot_Y = np.zeros((10, m))
        one_hot_Y[Y, np.arange(m)] = 1.
        return one_hot_Y
    
    def load_params(self, params_file_name):
        # params_3L_784x256x10.json
        with open(params_file_name, "r") as params_file:
            temp_params = json.load(params_file)
            for key in temp_params:
                self.params[key] = np.array(temp_params[key])
