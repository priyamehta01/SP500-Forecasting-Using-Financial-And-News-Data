from keras.layers import Layer
from keras import activations
import keras.backend as K

class get_attention(Layer):

    def __init__(self,
             no_of_cells=128,
             activation_func=None,
        **args):

        super(get_attention, self).__init__(**args)
        self.no_of_cells = no_of_cells
        self.activation_func = activations.get(activation_func)
        self._backend = K.backend()
        self.wei_x, self.wei_t, self.bxh = None, None, None
        self.wei_a, self.bxa = None, None

    def set_values(self):
        vals = {
            'no_of_cells': self.no_of_cells,
            'activation_func': activations.serialize(self.activation_func),
        }
        ini_values = super(get_attention, self).set_values()
        return dict(list(ini_values.items()) + list(vals.items()))

    def build(self, inp_shape):
        self.model_attention(inp_shape)
        super(get_attention, self).build(inp_shape)

    def call(self, inps, masking=None, **kwargs):
        store = self.model_emission(inps)
        store = self.activation_func(store)
        store = K.exp(store - K.max(store, axis=-1, keepdims=True))
        if masking is not None:
            masking = K.cast(masking, K.floatx())
            masking = K.expand_dims(masking)
            store = K.permute_dimensions(K.permute_dimensions(store * masking, (0, 2, 1)) * masking, (0, 2, 1))
        s_x = K.sum(store, axis=-1, keepdims=True)
        a_x = store / (s_x + K.epsilon())
        out = K.batch_dot(a_x, inps)
        return out

    def model_emission(self, inps):
        shape = K.shape(inps)
        batching, inp_len = shape[0], shape[1]
        store = K.expand_dims(K.dot(inps, self.wei_t), 2)
        k_x = K.expand_dims(K.dot(inps, self.wei_x), 1)
        h_x = K.tanh(store + k_x + self.bxh)
        out = K.reshape(K.dot(h_x, self.wei_a) + self.bxa, (batching, inp_len, inp_len))
        return out

    def model_attention(self, inp_shape):
        dimensions = int(inp_shape[2])

        self.wei_t = self.add_weight(shape=(dimensions, self.no_of_cells),
                                  name='{}_add_wei_t'.format(self.name),
                                  initializer='glorot_normal',
                                  regularizer=None,
                                  constraint=None)
        self.wei_x = self.add_weight(shape=(dimensions, self.no_of_cells),
                                  name='{}_add_wei_x'.format(self.name),
                                  initializer='glorot_normal',
                                  regularizer=None,
                                  constraint=None)
        self.bxh = self.add_weight(shape=(self.no_of_cells,),
                                  name='{}_add_bxh'.format(self.name),
                                  initializer='zeros',
                                  regularizer=None,
                                  constraint=None)

        self.wei_a = self.add_weight(shape=(self.no_of_cells, 1),
                                  name='{}_add_wei_a'.format(self.name),
                                  initializer='glorot_normal',
                                  regularizer=None,
                                  constraint=None)

        self.bxa = self.add_weight(shape=(1,),
                                  name='{}_add_bxa'.format(self.name),
                                  initializer='zeros',
                                  regularizer=None,
                                  constraint=None)
