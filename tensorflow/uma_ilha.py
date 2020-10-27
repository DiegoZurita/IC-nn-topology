import numpy as np
import tensorflow as tf
np.random.seed(123)
tf.random.set_seed(1234)


def gen_data():
    # Quadrado com canto superio esquerdo (-3,3) e inferior direito (3, -3)
    x_range = np.linspace(-3, 3, 30)
    #produto cartesiano
    xx, yy = np.meshgrid(x_range, x_range, indexing="ij")

    x = []
    y = []
    for i in range(len(xx)):
        for j in range(len(yy)):
            dist = np.power(xx[i,j], 2) + np.power(yy[i,j], 2)
            if dist <= 2:
                if dist <= 1.0:
                    y.append((0, 1))
                    x.append((xx[i, j], yy[i,j]))
            else:
                if np.random.uniform(0,1) >= 0.3: continue
                y.append((1,0))
                x.append((xx[i, j], yy[i,j]))

    return x, y

def main():
    x, y = gen_data()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        #tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(3, activation="relu"),
        tf.keras.layers.Dense(2, activation="tanh")
    ])

    #optim = tf.keras.optimizers.SGD(lr=0.471)
    optim = tf.keras.optimizers.Adam(learning_rate=0.031)
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    # model.summary()
    _ = model.fit(x, y, epochs=30)


if __name__ == "__main__":
    main()
