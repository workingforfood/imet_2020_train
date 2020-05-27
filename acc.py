import keras.backend as K
import tensorflow as tf

_EPSILON = 1e-7


def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)


# recall (true positive ratio to total positive)
def tp(y_true, y_pred):  # true_positive to target positive ratio
    # tp_vect = y_true * y_pred
    tp_count = K.sum(y_true * K.round(y_pred), axis=-1, keepdims=True)
    pos_count = K.sum(y_true, axis=-1, keepdims=True)
    return K.mean(tp_count / pos_count, axis=-1)


# false positive rate (false positive ratio to all predicted positive)
def fp(y_true, y_pred):  # false positive to predicted positive ratio
    # fp_vect = (_to_tensor(1., tf.float32) - y_true) * y_pred
    fp_count = K.sum((_to_tensor(1., tf.float32) - y_true) * K.round(y_pred), axis=-1, keepdims=True)
    pred_pos_count = K.clip(K.sum(K.round(y_pred), axis=-1, keepdims=True), min_value=1., max_value=3474.)
    return K.mean(fp_count / pred_pos_count, axis=-1)


if __name__ == "__main__":
    y_true = _to_tensor([[0., 1., 0., 0.], [0., 1., 1., 0.]], tf.float32)
    y_pred = _to_tensor([[0., 0.9, 0., 0.], [0., 0.7, 0., 0.9]], tf.float32)

    print(K.eval(tp(y_true, y_pred)))
    print(K.eval(fp(y_true, y_pred)))