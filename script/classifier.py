import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.utils import to_categorical


def load_and_split_data(path, ratio=0.8):
    data = np.load(path)
    obses, acts, states = data['tracks'], data['actions'], data['states']
    print(obses.shape, acts.shape, states.shape)
    assert len(obses) == len(acts)

    length = int(len(obses) * ratio)
    train_d, test_d = obses[:length], obses[length:]
    train_l, test_l = acts[:length], acts[length:]
    train_s, test_s = states[:length], states[length:]
    return train_d, train_l, test_d, test_l, train_s, test_s


model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(120, 6)),
    tf.keras.layers.LSTM(64, input_shape=(120, 6)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

# 载入数据集
x_train, y_train, x_test, y_test, s_train, s_test = load_and_split_data('..\\dataset\\real_trajectories.npz')

model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

print(y_train.shape)
model.fit(x_train, y_train, epochs=30,
          batch_size=128,
          validation_data=(x_test, y_test))

# todo: 添加指令执行的航空器
print(x_test.shape)
actions = []
for track in x_test:
    track = np.expand_dims(track, 0)
    action = model.predict(track).squeeze()
    action = np.concatenate([action, [0.0 for _ in range(26)]])
    actions.append(action)

actions = np.array(actions, dtype=np.float64)
print(s_test.shape, actions.shape)
un_matter = np.random.random((actions.shape[0], ))
np.savez('real_policy.npz',
         ept_rew=un_matter,
         obs=s_test,
         acs=actions,
         rews=un_matter,
         n_obs=s_test)
