import tensorflow as tf

class Initializer:
    @staticmethod
    # def parameter_initializer(shape,dtype='float32'):
    #     stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
    #     init = tf.random_uniform(shape,minval=-stdv,maxval=stdv,dtype=dtype,seed=1)
    #     return init

    def parameter_initializer(shape,dtype='float32'):
        # stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
        init = tf.random_uniform(shape,dtype=dtype)
        return init