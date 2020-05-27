import keras.backend as K
import tensorflow as tf

_EPSILON = 1e-7


# Type 1
def weighted_binary_crossentropy(y_true, y_pred, weight=2.):
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = 10. * (-(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log((1 - y_pred))))
    return K.mean(logloss, axis=-1)
########################################


def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)


# Type 4
# inspired by https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
# and https://arxiv.org/pdf/1708.02002.pdf
def binary_focal_loss(y_true, y_pred, gamma=2.):
    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    focall_loss = -(y_true * K.log(y_pred) * K.pow(1. - y_pred, gamma) +
                    (1 - y_true) * K.log(1 - y_pred) * K.pow(y_pred, gamma)) * 100.
    return K.mean(focall_loss, axis=-1)


if __name__ == "__main__":
    y_true = _to_tensor([[0., 1., 0., 0.], [0., 1., 1., 0.]], tf.float32)
    y_pred = _to_tensor([[0., 0.9, 0., 0.], [0., 1., 1., 0.]], tf.float32)

    print(K.eval(binary_focal_loss(y_true, y_pred)))