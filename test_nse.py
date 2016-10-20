from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
from keras import backend as K
import numpy
from nse import NSE, MultipleMemoryAccessNSE

input_val1 = numpy.random.randint(low=0, high=10, size=(1000, 10))
input_val2 = numpy.random.randint(low=0, high=10, size=(1000, 10))
target_val = numpy.random.rand(1000, 50)

input1 = Input(shape=(10,))
input2 = Input(shape=(10,))
embedding_for_nse = Embedding(input_dim=10, output_dim=50)
nse_encoder = NSE(50, return_mode="output_and_memory")

nse_embed_input1 = embedding_for_nse(input1)
nse_embed_input2 = embedding_for_nse(input2)  # (None, 10, 50)

nse_output = nse_encoder(nse_embed_input1)  # (None, 11, 50)

def make_input_for_mmanse(layer_outputs):
    nse_output_and_memory = layer_outputs[0]
    shared_memory = nse_output_and_memory[:, 1:, :]  # (None, 10, 50)
    mmanse_embed_input = layer_outputs[1]  # (None, 10, 50)
    return K.concatenate([mmanse_embed_input, shared_memory], axis=1)

get_output_shape = lambda input_shapes: (input_shapes[1][0], input_shapes[1][1]*2, input_shapes[1][2])
mmanse_input = Lambda(make_input_for_mmanse, output_shape=get_output_shape)([nse_output, nse_embed_input2])

mmanse_encoder = MultipleMemoryAccessNSE(50)
mmanse_output = mmanse_encoder(mmanse_input)

nse_model = Model(input=[input1, input2], output=mmanse_output)
nse_model.compile(optimizer='adam', loss='mse')

nse_model.summary()
nse_model.fit([input_val1, input_val2], target_val)
