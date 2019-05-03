from grpc.beta import implementations
import tensorflow as tf
import numpy
import pandas
from logistic_regression_input import *

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', '192.168.1.103:8500',
                           'inception_inference service host:port')
FLAGS = tf.app.flags.FLAGS


def main(_):
    x_tr,x_te,y_tr,y_te = input_data()
    x_test_arr = numpy.asarray(numpy.float32(x_te))
    #feed_value2 = numpy.asarray([90.234,1352.642,5978.735])
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'logistic_regression'
    request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
    #request.inputs['inputs'].float_val.append(feed_value2)
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_test_arr))
    request.output_filter.append('classes')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    prediction = stub.Predict(request, 5.0)  # 5 secs timeout
    print(prediction)
    

if __name__ == '__main__':
    tf.app.run()
