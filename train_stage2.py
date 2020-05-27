from networks import resnet50_classifier
from keras.optimizers import Adam
from train_datagenerator import DataGenerator
from networks import TestCallbackIters
from acc import tp, fp
from losses import binary_focal_loss
from keras.metrics import binary_accuracy

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = 100
    filepath = "chkpoints/resnet50_stage2_{iter}_{acc}_{tpacc}.h5"
    img_shape = (224, 224, 3)
    initial_epoch = 0
    num_of_classes = 3474

    train_gen = DataGenerator("/path/to/images/imet_collection_2020/train.csv",
                              "/path/to/images/imet_collection_2020/train",
                              # "/var/data/datasets/kaggle/imet_collection_2020/train",
                              batch_size=64, test=False, shuffle=True, test_split=0.1)

    test_gen = DataGenerator("/path/to/images/imet_collection_2020/train.csv",
                             "/path/to/images/imet_collection_2020/train",
                             # "/var/data/datasets/kaggle/imet_collection_2020/train",
                             batch_size=16, test=True, shuffle=False, test_split=0.1)

    model = resnet50_classifier(input_shape=img_shape, num_of_classes=num_of_classes)
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss=binary_focal_loss, metrics=[binary_accuracy, tp, fp])
    # load weights from best checkpoint of stage1
    model.load_weights("chkpoints/resnet50_stage1_38501_0.9990437134160651_0.5431706069356149.h5")

    model.summary()
    print("CONFIG:\n\tnum_of_classes: {}\n\tbatch_size: {}\n\tinitial_epoch: {}".format(num_of_classes, batch_size,
                                                                                        initial_epoch))

    checkpoint = TestCallbackIters(test_gen, 500, filepath, save_best=True)
    model.fit_generator(train_gen, use_multiprocessing=True, workers=4, epochs=10, max_queue_size=10,
                        validation_data=test_gen,
                        callbacks=[checkpoint], initial_epoch=initial_epoch)
