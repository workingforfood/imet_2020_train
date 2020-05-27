import os

from keras.applications import ResNet50
from keras.callbacks import Callback
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


class TestCallbackIters(Callback):
    def __init__(self, test_gen, iters, filename=None, save_best=False):
        super(TestCallbackIters, self).__init__()
        self.best_acc = 0.0
        self.best_tp_acc = 0.0
        self.current_iter = 1
        self.filename = filename
        self.filename_tpacc = filename
        self.iters = iters
        self.last_saved_filename = None
        self.last_saved_filename_tpacc = None
        self.save_best = save_best
        self.test_gen = test_gen

    def on_batch_end(self, batch, logs=None):
        if self.current_iter % self.iters == 0:
            loss, acc, tp_acc, fp_acc = self.model.evaluate_generator(self.test_gen)
            print('\nIter {} Testing loss: {}, acc: {}, tp acc: {}, fp acc: {}\n'.format(self.current_iter, loss, acc,
                                                                                         tp_acc, fp_acc))
            if self.filename is not None and (not self.save_best or self.best_acc < acc):
                save_fname = self.filename.format(acc=acc, iter=self.current_iter, tpacc=tp_acc, epoch="")
                self.model.save(save_fname)
                if self.last_saved_filename is not None and self.save_best:
                    os.remove(self.last_saved_filename)

                self.best_acc = acc
                self.last_saved_filename = save_fname

            if self.filename_tpacc is not None and (not self.save_best or self.best_tp_acc < tp_acc):
                save_fname = self.filename_tpacc.format(acc=acc, iter=self.current_iter+1, tpacc=tp_acc, epoch="")
                self.model.save(save_fname)
                if self.last_saved_filename_tpacc is not None and self.save_best:
                    os.remove(self.last_saved_filename_tpacc)

                self.best_tp_acc = tp_acc
                self.last_saved_filename_tpacc = save_fname

        self.current_iter += 1


def resnet50_classifier(num_of_classes, input_shape=(224, 224, 3)):
    input_tensor = Input(shape=input_shape)
    resnet50_model = ResNet50(input_tensor=input_tensor, input_shape=input_shape, weights="imagenet",
                              include_top=False)
    x = resnet50_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.5)(x)
    encoded = Dense(num_of_classes, activation='sigmoid', kernel_initializer='he_normal')(x)
    return Model(inputs=input_tensor, outputs=encoded, name="Resnet50_shape{}_classes{}".format(input_shape,
                                                                                                num_of_classes))
