import keras.backend as K
from networks import resnet50_classifier
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def keras_to_tensorflow(keras_model, output_dir, model_name):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []
    for i in range(len(keras_model.inputs)):
        print(keras_model.inputs[i].name)

    for i in range(len(keras_model.outputs)):
        out_nodes.append(keras_model.outputs[i].name.replace(":0", ""))
        print(keras_model.outputs[i].name.replace(":0", ""))

    sess = K.get_session()

    from tensorflow.python.framework import graph_util, graph_io

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)


filename = "chkpoints/resnet50_stage2_8000_0.9991412270847742_0.5025544129983261.h5"
save_filename = "{}.pb".format(filename.split("/")[-1].replace(".h5", ""))
alpha = 1.0
img_shape = (224, 224, 3)
num_of_classes = 3474

K.set_learning_phase(0)
model = resnet50_classifier(input_shape=img_shape, num_of_classes=num_of_classes)
model.load_weights(filename)
keras_to_tensorflow(keras_model=model, output_dir="chkpoints", model_name=save_filename)

