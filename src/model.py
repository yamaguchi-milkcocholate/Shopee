import tensorflow as tf
import efficientnet.tfkeras as efn
from classification_models.keras import Classifiers
import math


# Arcmarginproduct class keras layer
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# Function to build bert model
def build_bert_model(bert_layer, n_classes, lr, max_len = 512, train=True, emb_len=None):
    
    margin = ArcMarginProduct(
            n_classes = n_classes, 
            s = 30, 
            m = 0.5, 
            name='head/arc_margin', 
            dtype='float32'
            )
    
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    label = tf.keras.layers.Input(shape = (), name = 'label')

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    if emb_len is not None:
        x = tf.keras.layers.Dropout(0.2)(clf_output)
        x = tf.keras.layers.Dense(emb_len)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = margin([x, label])
    output = tf.keras.layers.Softmax(dtype='float32')(x)
    model = tf.keras.models.Model(inputs = [input_word_ids, input_mask, segment_ids, label], outputs = [output])
    if compile:
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = lr),
                    loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
                    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def build_resnext_model(n_classes, image_size, lr, train=True, emb_len=None):
    margin = ArcMarginProduct(
        n_classes = n_classes, 
        s = 30, 
        m = 0.5, 
        name='head/arc_margin', 
        dtype='float32'
    )

    inp = tf.keras.layers.Input(shape = (*image_size, 3), name = 'inp1')
    label = tf.keras.layers.Input(shape = (), name = 'inp2')
    ResNext, _ = Classifiers.get('resnext50')
    x = ResNext(input_shape=(*image_size, 3), weights = 'imagenet', include_top = False)(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if emb_len is not None:
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(emb_len)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = margin([x, label])

    output = tf.keras.layers.Softmax(dtype='float32')(x)

    model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])

    opt = tf.keras.optimizers.Adam(learning_rate = lr)

    model.compile(
        optimizer = opt,
        loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    ) 

    return model


def build_efficientnet_model(n_classes, image_size, lr, en_type='B0', train=True, emb_len=None):
    margin = ArcMarginProduct(
        n_classes = n_classes, 
        s = 30, 
        m = 0.5, 
        name='head/arc_margin', 
        dtype='float32'
    )

    inp = tf.keras.layers.Input(shape = (*image_size, 3), name = 'inp1')
    label = tf.keras.layers.Input(shape = (), name = 'inp2')
    x = getattr(efn, f'EfficientNet{en_type}')(weights = 'imagenet', include_top = False)(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if emb_len is not None:
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(emb_len)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = margin([x, label])

    output = tf.keras.layers.Softmax(dtype='float32')(x)

    model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])

    opt = tf.keras.optimizers.Adam(learning_rate = lr)

    model.compile(
        optimizer = opt,
        loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    ) 

    return model