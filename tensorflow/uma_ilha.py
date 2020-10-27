import numpy as np
import tensorflow as tf
np.random.seed(123)
tf.random.set_seed(1234)


def gen_data(n=30):
    # Quadrado com canto superio esquerdo (-3,3) e inferior direito (3, -3)
    x_range = np.linspace(-3, 3, n)
    #produto cartesiano
    xx, yy = np.meshgrid(x_range, x_range, indexing="ij")

    x = []
    y = []

    c1 = 0
    c2 = 0
    for i in range(len(xx)):
        for j in range(len(yy)):
            dist = np.power(xx[i,j], 2) + np.power(yy[i,j], 2)
            if dist <= 2:
                if dist <= 1.3:
                    y.append((0, 1))
                    x.append((xx[i, j], yy[i,j]))
                    c1+=1
            else:
                if np.random.uniform(0,1) >= 0.4: continue
                y.append((1,0))
                x.append((xx[i, j], yy[i,j]))
                c2+=1

    

    return x, y, c1, c2, len(x)

def main():
    x, y, n_c1, n_c2, n = gen_data(30)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        #tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(
            units=3, 
            activation="relu",
            bias_initializer="random_uniform",
            name="internal"
        ),
        tf.keras.layers.Dense(
            units=2, 
            activation="tanh", 
            bias_initializer="random_uniform",
            name="output"
        )
    ])

    optim = tf.keras.optimizers.Adam(learning_rate=0.031)
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    # model.summary()
    _ = model.fit(x, y, epochs=40)

    print("")
    print("n: {}".format(n))
    print("c1: {:.2f}".format(n_c1/n))
    print("c2: {:.2f}".format(n_c2/n))
    for layer in model.layers:
        print("----------- Internal layer: {} -----------".format(layer.name))
        print("Weghts:")
        print(layer.weights[0].numpy())
        print("Bias:")
        print(layer.bias.numpy())
        print(" ")


if __name__ == "__main__":
    main()
