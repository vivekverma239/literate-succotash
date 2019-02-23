import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import tensorflow_hub as hub

class Match(Layer):
    """Layer that computes a matching matrix between samples in two tensors.
    # Arguments
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, normalize=False, match_type='dot', **kwargs):
        super(Match, self).__init__(**kwargs)
        self.normalize = normalize
        self.match_type = match_type
        self.supports_masking = True
        if match_type not in ['dot', 'mul', 'plus', 'minus', 'concat']:
            raise ValueError('In `Match` layer, '
                             'param match_type=%s is unknown.' % match_type)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Match` layer should be called '
                             'on a list of 2 inputs.')
        self.shape1 = input_shape[0]
        self.shape2 = input_shape[1]
        if self.shape1[0] != self.shape2[0]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (self.shape1[0], self.shape2[0]) +
                'Layer shapes: %s, %s' % (self.shape1, self.shape2))
        if self.shape1[2] != self.shape2[2]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (self.shape1[2], self.shape2[2]) +
                'Layer shapes: %s, %s' % (self.shape1, self.shape2))

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        if self.match_type in ['dot']:
            if self.normalize:
                x1 = K.l2_normalize(x1, axis=2)
                x2 = K.l2_normalize(x2, axis=2)
            output = K.batch_dot(x1, x2, axes=(2,2))
            output = K.expand_dims(output, 3)
        elif self.match_type in ['mul', 'plus', 'minus']:
            x1_exp = K.stack([x1] * self.shape2[1], 2)
            x2_exp = K.stack([x2] * self.shape1[1], 1)
            if self.match_type == 'mul':
                output = x1_exp * x2_exp
            elif self.match_type == 'plus':
                output = x1_exp + x2_exp
            elif self.match_type == 'minus':
                output = x1_exp - x2_exp
        elif self.match_type in ['concat']:
            x1_exp = K.stack([x1] * self.shape2[1], axis=2)
            x2_exp = K.stack([x2] * self.shape1[1], axis=1)
            output = K.concat([x1_exp, x2_exp], axis=3)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Match` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError('A `Match` layer should be called '
                             'on 2 inputs with 3 dimensions.')
        if shape1[0] != shape2[0] or shape1[2] != shape2[2]:
            raise ValueError('A `Match` layer should be called '
                             'on 2 inputs with same 0,2 dimensions.')

        if self.match_type in ['dot']:
            output_shape = [shape1[0], shape1[1], shape2[1], 1]
        elif self.match_type in ['mul', 'plus', 'minus']:
            output_shape = [shape1[0], shape1[1], shape2[1], shape1[2]]
        elif self.match_type in ['concat']:
            output_shape = [shape1[0], shape1[1], shape2[1], shape1[2]+shape2[2]]

        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'normalize': self.normalize,
            'match_type': self.match_type,
        }
        base_config = super(Match, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def match(inputs, axes, normalize=False, match_type='dot', **kwargs):
    """Functional interface to the `Match` layer.
    # Arguments
        inputs: A list of input tensors (with exact 2 tensors).
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the dot product matching matrix of the samples
        from the inputs.
    """
    return Match(normalize=normalize, match_type=match_type, **kwargs)(inputs)


class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable=False, **kwargs):
        self.dimensions = 1024
        self.trainable = trainable
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights.extend(tf.trainable_variables(scope="^{}_module/.*".format(self.name)))
        sess = tf.Session()
        K.set_session(sess)
        # Initialize sessions
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        seq_length = self.compute_length(x)
        seq_length = K.stop_gradient(seq_length)
        inputs={
                    "tokens": x,
                    "sequence_len": seq_length
                }
        result = self.elmo(inputs=inputs,
                      as_dict=True,
                      signature="tokens"
                      )['elmo']
        input_shape = x.get_shape()
        result = K.reshape(result, [-1,\
                                    int(input_shape[1]),\
                                    int(result.get_shape()[-1])])
        return result

    def compute_length(self, inputs):
        mask = K.not_equal(inputs, '--PAD--')
        mask = K.cast(mask, tf.int32)
        seq_length = K.sum(mask, axis=-1)
        return seq_length

    # def compute_mask(self, inputs, mask=None):
    #     return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dimensions)

if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    import numpy as np
    b = Input(batch_shape=(None, 5), dtype=tf.string)
    a = ElmoEmbeddingLayer()(b  )
    model = Model(inputs=b, outputs=a)
    model.compile("adam", "categorical_crossentropy")
    input_ = np.array([["This", "is", "not", "good", "--PAD--"]])
    print(model.predict(input_))
