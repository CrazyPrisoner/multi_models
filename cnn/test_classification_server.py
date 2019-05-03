from grpc.beta import implementations
import tensorflow as tf
import numpy
import pandas
from data_processing import *

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', '192.168.1.103:8500',
                           'inception_inference service host:port')
FLAGS = tf.app.flags.FLAGS


def main(_):
    # Import correct data from script
    training_images = train_data_with_label()
    testing_images = test_data_with_label()
    tr_img_data = numpy.array([i[0] for i in training_images])
    tr_lbl_data = numpy.array([i[1] for i in training_images])
    tst_img_data = numpy.array([i[0] for i in testing_images])
    tst_lbl_data = numpy.array([i[1] for i in testing_images])
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'cnn'
    request.model_spec.signature_name = 'predict'
    request.inputs['input'].dtype = types_pb2.DT_INT32
    #request.inputs['inputs'].float_val.append(feed_value2)
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(tr_img_data[0].astype(dtype=numpy.float32)))
    request.inputs['prob'].CopyFrom(tf.contrib.util.make_tensor_proto(0.8))
    request.output_filter.append('output')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    prediction = stub.Predict(request, 5.0)  # 5 secs timeout
    floats = prediction.outputs['output'].float_val
    pred_arr = numpy.array(floats)
    pred_arr = pred_arr.reshape(-1, 5)
    #pred_df = pandas.DataFrame(columns = ['normal', 'pb'], data=pred_arr)
    print(prediction)


if __name__ == '__main__':
    tf.app.run()
