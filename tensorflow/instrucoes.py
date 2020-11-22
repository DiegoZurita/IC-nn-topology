import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def gen_data(n):
    X = [ ]
    y = []
    interval = np.linspace(-2, 2, n)

    for i in range(n):
        X.append( (-1, interval[i]) )
        y.append(-1)
        X.append( (1, interval[i]) )
        y.append(1)

    return np.array(X), np.array(y)

def main():
    ## Gera os dados conforme o pdf
    X, y = gen_data(10)

    ## Visualiza os dados
    # plt.title("Dados gerados")
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    ## Montamos uma rede com a unica camada.
    only_layer = tf.keras.layers.Dense(1)

    model = tf.keras.models.Sequential([
        only_layer
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss='mean_squared_error',
        metrics=["mae"]
    )

    ## Treino da rede
    model.fit(X, y, epochs=100, batch_size=2, verbose=0)

    print("Pesos pós treino")
    print(only_layer.weights[0].numpy())
    print("Bias pós treino")
    print(only_layer.weights[1].numpy())
    print("")
    print("Predicts")
    print(model.predict(X))
    print("Real")
    print(y)


    ## ------- Minimizando -----------
    ## Aqui vou minimizar na "unha"
    weights = tf.Variable( np.random.random(size=(2, )), dtype=tf.float32 )
    bias = tf.Variable( np.random.random(size=1), dtype=tf.float32 )
    
    ## Definição da função loss
    def loss_fn():
        n = len(X)
        s = 0
        for i in range(n):
            x = X[i]
            _y = y[i]
            b = weights[0]*x[0] + weights[1]*x[1] + bias
            s += tf.math.pow( _y - b , 2 )

        return s/(2*n)

    print()
    print('Pesos e bias antes da minimização')
    print(weights.numpy(), bias.numpy())

    print("minimizando..")
    opt = tf.optimizers.SGD()
    n_iteracoes = 500
    for _ in range(n_iteracoes):
        opt.minimize(loss=loss_fn, var_list=[weights, bias])
        #print("Loss: ", loss_fn().numpy())

    print("Iterações:", opt.iterations.numpy())
    print('Pesos e bias depois da minimização')
    print(weights.numpy(), bias.numpy())
    print("Loss: ", loss_fn().numpy())


if __name__ == "__main__":
    main()