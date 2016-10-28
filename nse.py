import sys
from overrides import overrides

from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.layers import LSTM, Dense

class NSE(Layer):
    '''
    Simple Neural Semantic Encoder.
    '''
    def __init__(self, output_dim, input_length=None, composer_activation='linear',
                 return_mode='last_output', weights=None, **kwargs):
        '''
        Arguments:
        output_dim (int)
        input_length (int)
        composer_activation (str): activation used in the MLP
        return_mode (str): One of last_output, all_outputs, output_and_memory
            This is analogous to the return_sequences flag in Keras' Recurrent.
            last_output returns only the last h_t
            all_outputs returns the whole sequence of h_ts
            output_and_memory returns the last output and the last memory concatenated
                (needed if this layer is followed by a MMA-NSE)
        weights (list): Initial weights
        '''
        self.output_dim = output_dim
        self.input_dim = output_dim  # Equation 2 in the paper makes this assumption.
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]
        self.input_length = input_length
        self.composer_activation = composer_activation
        super(NSE, self).__init__(**kwargs)
        self.reader = LSTM(self.output_dim, return_sequences=True, name="{}_reader".format(self.name))
        # TODO: Let the writer use parameter dropout and any consume_less mode.
        # Setting dropout to 0 here to eliminate the need for constants.
        # Setting consume_less to mem to eliminate need for preprocessing
        self.writer = LSTM(self.output_dim, dropout_W=0.0, dropout_U=0.0, consume_less="mem",
                           name="{}_writer".format(self.name))
        self.composer = Dense(self.output_dim * 2, activation=self.composer_activation,
                              name="{}_composer".format(self.name))
        if return_mode not in ["last_output", "all_outputs", "output_and_memory"]:
            raise Exception("Unrecognized return mode: %s" % (return_mode))
        self.return_mode = return_mode

    def get_output_shape_for(self, input_shape):
        input_length = input_shape[1]
        if self.return_mode == "last_output": 
            return (input_shape[0], self.output_dim)
        elif self.return_mode == "all_outputs":
            return (input_shape[0], input_length, self.output_dim)
        else:
            # return_mode is output_and_memory. Output will be concatenated to memory.
            return (input_shape[0], input_length + 1, self.output_dim)

    def compute_mask(self, input, mask):
        if mask is None or self.return_mode == "last_output":
            return None
        elif self.return_mode == "all_outputs":
            return mask  # (batch_size, input_length)
        else:
            # Return mode is output_and_memory
            # Mask memory corresponding to all the inputs that are masked, and do not mask the output
            # (batch_size, input_length + 1)
            return K.cast(K.concatenate([K.zeros_like(mask[:, :1]), mask]), 'uint8')

    def get_composer_input_shape(self, input_shape):
        # Takes concatenation of output and memory summary
        return (input_shape[0], self.output_dim * 2)

    def get_reader_input_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[-1]
        assert self.reader.return_sequences, "The reader has to return sequences!"
        reader_input_shape = self.get_reader_input_shape(input_shape)
        print >>sys.stderr, "NSE reader input shape:", reader_input_shape 
        writer_input_shape = (input_shape[0], 1, self.output_dim * 2)  # Will process one timestep at a time
        print >>sys.stderr, "NSE writer input shape:", writer_input_shape 
        composer_input_shape = self.get_composer_input_shape(input_shape)
        print >>sys.stderr, "NSE composer input shape:", composer_input_shape 
        self.reader.build(reader_input_shape)
        self.writer.build(writer_input_shape)
        self.composer.build(composer_input_shape)

        # Aggregate weights of individual components for this layer.
        reader_weights = self.reader.trainable_weights
        writer_weights = self.writer.trainable_weights
        composer_weights = self.composer.trainable_weights
        self.trainable_weights = reader_weights + writer_weights + composer_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def read(self, nse_input, input_mask=None):
        '''
        This method produces the 'read' output (equation 1 in the paper) for all timesteps
        and initializes the memory slot mem_0.

        Input: nse_input (batch_size, input_length, input_dim)
        Outputs:
            o (batch_size, input_length, output_dim)
            flattened_mem_0 (batch_size, input_length * output_dim)
 
        While this method simply copies input to mem_0, variants that inherit from this class can do
        something fancier.
        '''
        input_to_read = nse_input
        mem_0 = input_to_read
        flattened_mem_0 = K.batch_flatten(mem_0)
        o = self.reader.call(input_to_read, input_mask)
        o_mask = self.reader.compute_mask(input_to_read, input_mask)
        return o, [flattened_mem_0], o_mask

    @staticmethod
    def summarize_memory(o_t, mem_tm1):
        '''
        This method selects the relevant parts of the memory given the read output and summarizes the
        memory. Implements Equations 2-3 or 8-11 in the paper.
        '''
        # Selecting relevant memory slots, Equation 2
        z_t = K.softmax(K.sum(K.expand_dims(o_t, dim=1) * mem_tm1, axis=2))  # (batch_size, input_length)
        # Summarizing memory, Equation 3
        m_rt = K.sum(K.expand_dims(z_t, dim=2) * mem_tm1, axis=1)  # (batch_size, output_dim)
        return z_t, m_rt

    def compose_memory_and_output(self, output_memory_list):
        '''
        This method takes a list of tensors and applies the composition function on their concatrnation.
        Implements equation 4 or 12 in the paper.
        '''
        # Composition, Equation 4
        c_t = self.composer.call(K.concatenate(output_memory_list))  # (batch_size, output_dim)
        return c_t

    def update_memory(self, z_t, h_t, mem_tm1):
        '''
        This method takes the attention vector (z_t), writer output (h_t) and previous timestep's memory (mem_tm1)
        and updates the memory. Implements equations 6, 14 or 15.
        '''
        tiled_z_t = K.tile(K.expand_dims(z_t), (self.output_dim))  # (batch_size, input_length, output_dim)
        input_length = K.shape(mem_tm1)[1]
        # (batch_size, input_length, output_dim)
        tiled_h_t = K.permute_dimensions(K.tile(K.expand_dims(h_t), (input_length)), (0, 2, 1))
        # Updating memory. First term in summation corresponds to selective forgetting and the second term to
        # selective addition. Equation 6.
        mem_t = mem_tm1 * (1 - tiled_z_t) + tiled_h_t * tiled_z_t  # (batch_size, input_length, output_dim)
        return mem_t

    def compose_and_write_step(self, o_t, states):
        '''
        This method is a step function that updates the memory at each time step and produces
        a new output vector (Equations 2 to 6 in the paper).
        The memory_state is flattened because K.rnn requires all states to be of the same shape as the output,
        because it uses the same mask for the output and the states.
        Inputs:
            o_t (batch_size, output_dim)
            states (list[Tensor])
                flattened_mem_tm1 (batch_size, input_length * output_dim)
                writer_h_tm1 (batch_size, output_dim)
                writer_c_tm1 (batch_size, output_dim)

        Outputs:
            h_t (batch_size, output_dim)
            flattened_mem_t (batch_size, input_length * output_dim)
        '''
        flattened_mem_tm1, writer_h_tm1, writer_c_tm1 = states
        input_mem_shape = K.shape(flattened_mem_tm1)
        mem_tm1_shape = (input_mem_shape[0], input_mem_shape[1]/self.output_dim, self.output_dim)
        mem_tm1 = K.reshape(flattened_mem_tm1, mem_tm1_shape)  # (batch_size, input_length, output_dim)
        z_t, m_rt = self.summarize_memory(o_t, mem_tm1)
        c_t = self.compose_memory_and_output([o_t, m_rt])
        # Collecting the necessary variables to directly call writer's step function.
        writer_constants = self.writer.get_constants(c_t)  # returns dropouts for W and U (all 1s, see init)
        writer_states = [writer_h_tm1, writer_c_tm1] + writer_constants
        # Making a call to writer's step function, Equation 5
        h_t, [_, writer_c_t] = self.writer.step(c_t, writer_states)  # h_t, writer_c_t: (batch_size, output_dim)
        mem_t = self.update_memory(z_t, h_t, mem_tm1)
        flattened_mem_t = K.batch_flatten(mem_t)
        return h_t, [flattened_mem_t, h_t, writer_c_t]

    def call(self, x, mask=None):
        # input_shape = (batch_size, input_length, input_dim). This needs to be defined in build.
        read_output, initial_memory_states, output_mask = self.read(x, mask)
        initial_write_states = self.writer.get_initial_states(read_output)  # h_0 and c_0 of the writer LSTM
        initial_states = initial_memory_states + initial_write_states
        # last_output: (batch_size, output_dim)
        # all_outputs: (batch_size, input_length, output_dim)
        # last_states:
        #       last_memory_state: (batch_size, input_length, output_dim)
        #       last_output
        #       last_writer_ct
        last_output, all_outputs, last_states = K.rnn(self.compose_and_write_step, read_output, initial_states,
                                                      mask=output_mask)
        last_memory = last_states[0]
        if self.return_mode == "last_output":
            return last_output
        elif self.return_mode == "all_outputs":
            return all_outputs
        else:
            # return mode is output_and_memory
            expanded_last_output = K.expand_dims(last_output, dim=1)  # (batch_size, 1, output_dim)
            # (batch_size, 1+input_length, output_dim)
            return K.concatenate([expanded_last_output, last_memory], axis=1)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_length': self.input_length,
                  'composer_activation': self.composer_activation,
                  'return_mode': self.return_mode}
        base_config = super(NSE, self).get_config()
        config.update(base_config)
        return config

class MultipleMemoryAccessNSE(NSE):
    '''
    MultipleMemoryAccessNSE is very similar to the simple NSE. The difference is that along with the sentence
    memory, it has access to one (or multiple) additional memory. The operations on the additional memory are
    exactly the same as the original memory. The additional memory is initialized from the final timestep of
    a different NSE, and the composer will take as input the concatenation of the reader output and summaries
    of both the memories.
    '''
    #TODO: This is currently assuming we need access to one additional memory. Change it to an arbitrary number.
    @overrides
    def get_output_shape_for(self, input_shape):
        # This class has twice the input length as an NSE due to the concatenated input. Pass the right size
        # to NSE's method to get the right putput shape.
        nse_input_shape = (input_shape[0], input_shape[1]/2, input_shape[2])
        return super(MultipleMemoryAccessNSE, self).get_output_shape_for(nse_input_shape)

    def get_reader_input_shape(self, input_shape):
        return (input_shape[0], input_shape[1]/2, self.output_dim)

    def get_composer_input_shape(self, input_shape):
        return (input_shape[0], self.output_dim * 3)

    @overrides
    def read(self, nse_input, input_mask=None):
        '''
        Read input in MMA-NSE will be of shape (batch_size, read_input_length*2, input_dim), a concatenation of
        the actual input to this NSE and the output from a different NSE. The latter will be used to initialize
        the shared memory. The former will be passed to the read LSTM and also used to initialize the current
        memory.
        '''
        input_length = K.shape(nse_input)[1]
        read_input_length = input_length/2
        input_to_read = nse_input[:, :read_input_length, :]
        initial_shared_memory = K.batch_flatten(nse_input[:, read_input_length:, :])
        mem_0 = K.batch_flatten(input_to_read)
        o = self.reader.call(input_to_read, input_mask)
        o_mask = self.reader.compute_mask(input_to_read, input_mask)
        return o, [mem_0, initial_shared_memory], o_mask

    @overrides
    def compose_and_write_step(self, o_t, states):
        flattened_mem_tm1, flattened_shared_mem_tm1, writer_h_tm1, writer_c_tm1 = states
        input_mem_shape = K.shape(flattened_mem_tm1)
        mem_shape = (input_mem_shape[0], input_mem_shape[1]/self.output_dim, self.output_dim)
        mem_tm1 = K.reshape(flattened_mem_tm1, mem_shape)
        shared_mem_tm1 = K.reshape(flattened_shared_mem_tm1, mem_shape)
        z_t, m_rt = self.summarize_memory(o_t, mem_tm1)
        shared_z_t, shared_m_rt = self.summarize_memory(o_t, shared_mem_tm1)
        c_t = self.compose_memory_and_output([o_t, m_rt, shared_m_rt])
        # Collecting the necessary variables to directly call writer's step function.
        writer_constants = self.writer.get_constants(c_t)  # returns dropouts for W and U (all 1s, see init)
        writer_states = [writer_h_tm1, writer_c_tm1] + writer_constants
        # Making a call to writer's step function, Equation 5
        h_t, [_, writer_c_t] = self.writer.step(c_t, writer_states)  # h_t, writer_c_t: (batch_size, output_dim)
        mem_t = self.update_memory(z_t, h_t, mem_tm1)
        shared_mem_t = self.update_memory(shared_z_t, h_t, shared_mem_tm1)
        return h_t, [K.batch_flatten(mem_t), K.batch_flatten(shared_mem_t), h_t, writer_c_t]


class InputMemoryMerger(Layer):
    '''
    This layer taks as input, the memory part of the output of a NSE layer, and the embedded input to a MMANSE
    layer, and prepares a single input tensor for MMANSE that is a concatenation of the first sentence's memory
    and the second sentence's embedding.
    This is a concrete layer instead of a lambda function because we want to support masking.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(InputMemoryMerger, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1]*2, input_shapes[1][2])

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        elif mask == [None, None]:
            return None
        else:
            memory_mask, mmanse_embed_mask = mask
            return K.concatenate([mmanse_embed_mask, memory_mask], axis=1)  # (batch_size, nse_input_length * 2)
        
    def call(self, inputs, mask=None):
        shared_memory = inputs[0]
        mmanse_embed_input = inputs[1]  # (batch_size, nse_input_length, output_dim)
        return K.concatenate([mmanse_embed_input, shared_memory], axis=1)

class OutputSplitter(Layer):
    '''
    This layer takes the concatenation of output and memory from NSE and returns either the output or the
    memory.
    '''
    def __init__(self, return_mode, **kwargs):
        self.supperots_masking = True
        if return_mode not in ["output", "memory"]:
            raise Exception("Invalid return mode: %s" % return_mode)
        self.return_mode = return_mode
        super(OutputSplitter, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_mode == "output":
            return (input_shape[0], input_shape[2])
        else:
            # Return mode is memory.
            # input contains output and memory concatenated along the second dimension.
            return (input_shape[0], input_shape[1] - 1, input_shape[2])

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if self.return_mode == "output" or mask is None:
            return None
        else:
            # Return mode is memory and mask is not None
            return mask[:, 1:]  # (batch_size, nse_input_length)

    def call(self, inputs, mask=None):
        if self.return_mode == "output":
            return inputs[:, 0, :]  # (batch_size, output_dim)
        else:
            return inputs[:, 1:, :]  # (batch_size, nse_input_length, output_dim)
