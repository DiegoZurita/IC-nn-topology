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

    #optim = tf.keras.optimizers.SGD(lr=0.471)
    optim = tf.keras.optimizers.Adam(learning_rate=0.031)
    model.compile(
        optimizer=optim,
        loss='mean_squared_error',
        metrics=["accuracy"]
    )

    # model.summary()
    _ = model.fit(x, y, epochs=30)

    print("")
    for layer in model.layers:
        print("-----------Internal layer {}-----------".format(layer.name))
        print("Weghts")
        print(layer.weights[0].numpy())
        print("bias")
        print(layer.bias.numpy())
        print(" ")


if __name__ == "__main__":
    main()
