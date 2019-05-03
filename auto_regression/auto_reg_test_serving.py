import numpy
import pandas
import json
import requests
import tensorflow as tf
from linear_input import  *

from grpc.beta import implementations
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', '192.168.1.103:8500',
                           'inception_inference service host:port')
FLAGS = tf.app.flags.FLAGS

def main(_):
    x, train_x,test_x = input_data()
    # Wrap bitstring in JSON
    out_pp = numpy.array(train_x[-1])
    strafe = out_pp.astype(numpy.float32)

 # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'auto_regression'
    request.model_spec.signature_name = 'predict_value'
    request.inputs['input_value'].dtype = types_pb2.DT_INT32
    #request.inputs['inputs'].float_val.append(feed_value2)
    request.inputs['input_value'].CopyFrom(
        tf.contrib.util.make_tensor_proto(test_x.astype(dtype=numpy.float32)))
    request.output_filter.append('output_value')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    prediction = stub.Predict(request, 5.0)  # 5 secs timeout
    floats = prediction.outputs['output_value'].float_val
    print(prediction)

if __name__ == '__main__':
    tf.app.run()


