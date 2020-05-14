import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Layer
from tensorflow.keras.models import Model
import numpy as np
import os
import tensorflow.keras.backend as K


class VEmbeddingAutoEncoder:
    def __init__(self, input_set, D, weights_path, vs_len, encoding_dim=3, batch_size=50, emb_alpha=0.1):
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.x = input_set
        self.input_shape = len(input_set[0])
        self.D = D
        self.vs_len = vs_len
        self.aa_shape = self.input_shape - self.vs_len
        self.lambda1 = self.create_lambda1()
        self.lambda2 = self.create_lambda2()
        self.weights_path = weights_path
        # embedding
        self.emb_alpha = emb_alpha
        print(self.x)

    def create_lambda1(self):
        return self.MyLambda1(self.vs_len, self.aa_shape)

    def create_lambda2(self):
        return self.MyLambda2(self.vs_len)

    class MyLambda1(Layer):
        # this class is Lambda layer for the CDR3 input one hot split
        # exactly as writing instead: decoded3_0 = Lambda(lambda x: x[:, :-self.vs_len])(x)
        def __init__(self, vs_n, aa_n, **kwargs):
            self.vs_n = vs_n
            self.aa_n = aa_n
            super(VEmbeddingAutoEncoder.MyLambda1, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VEmbeddingAutoEncoder.MyLambda1, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, :-self.vs_n]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.aa_n)

        def get_config(self):
            config = super(VEmbeddingAutoEncoder.MyLambda1, self).get_config()
            config['vs_n'] = self.vs_n
            config['aa_n'] = self.aa_n
            return config

    class MyLambda2(Layer):
        # this class is Lambda layer for the Vs input one hot split
        # exactly as writing instead: decoded3_1 = Lambda(lambda x: x[:, -self.vs_len:])(x)
        def __init__(self, vs_n, **kwargs):
            self.vs_n = vs_n
            super(VEmbeddingAutoEncoder.MyLambda2, self).__init__(**kwargs)

        def build(self, input_shape):
            super(VEmbeddingAutoEncoder.MyLambda2, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return x[:, -self.vs_n:]

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.vs_n)

        def get_config(self):
            config = super(VEmbeddingAutoEncoder.MyLambda2, self).get_config()
            config['vs_n'] = self.vs_n
            return config

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        print(model.summary())
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        x = Dense(self.input_shape, activation='elu')(dropout2)
        # split the softmax layer for the 'aa one-hots' and 'v one-hot'
        decoded3_0 = self.lambda1(x)
        decoded3_1 = self.lambda2(x)
        reshape_0 = Reshape((int(self.aa_shape / 21), 21))(decoded3_0)
        reshape_1 = Reshape((1, self.vs_len))(decoded3_1)
        decoded4_0 = Dense(21, activation='softmax')(reshape_0)
        decoded4_1 = Dense(self.vs_len, activation='softmax')(reshape_1)
        reshape2_0 = Reshape((self.aa_shape,))(decoded4_0)
        reshape2_1 = Reshape((self.vs_len,))(decoded4_1)
        both = tensorflow.keras.layers.concatenate([reshape2_0, reshape2_1], axis=1)
        # ---
        model = Model(inputs, both)
        print(model.summary())
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        # create the model using our input (cdr3 sequences set) and
        # two separate outputs -- one for the reconstruction of the
        # data and another for the representations, respectively
        model = Model(inputs=inputs, outputs=[dc_out, ec_out])
        self.model = model
        return model

    # 1. y_true 2. y_pred
    def vae_loss(self, D, ec_out):
        emb = 0
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                if i < j:
                    # norm 2
                    dis_z = K.sqrt(K.sum(K.square(ec_out[i] - ec_out[j])))
                    # norm 1
                    # dis_z = K.sum(K.abs(ec_out[i] - ec_out[j]))
                    emb += K.square(dis_z - D[i][j])
        emb = self.emb_alpha * emb
        return emb

    def generator(self, X, D, batch):
        # generate: A. batches of x B. distances matrix within batch
        while True:
            inds = []
            for i in range(batch):
                # choose random index in features
                index = np.random.choice(X.shape[0], 1)[0]
                if i == 0:
                    batch_X = np.array([X[index]])
                else:
                    batch_X = np.concatenate((batch_X, np.array([X[index]])), axis=0)
                inds.append(index)
            tmp = D[np.array(inds)]
            batch_D = tmp[:, np.array(inds)]
            # 1. training data-features 2. target data-labels
            yield batch_X, [batch_X, batch_D]

    def fit_generator(self, epochs=300):
        # self.model = multi_gpu_model(self.model, gpus=3)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss=['mse', self.vae_loss])
        results = self.model.fit_generator(self.generator(self.x, self.D, self.batch_size),
                                           steps_per_epoch=self.x.shape[0] / self.batch_size,
                                           epochs=epochs, verbose=2)
        return results

    def fit_autoencoder(self, train_x, batch_size=10, epochs=300):
        # self.model = multi_gpu_model(self.model, gpus=3)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss='mse')
        log_dir = './log/'
        tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')
        results = self.model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                 epochs=epochs, batch_size=batch_size,
                                 callbacks=[tb_callback, es_callback])
        return results

    def save_ae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/v_embedding_encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/v_embedding_decoder_weights.h5')
        self.model.save(r'./weights_' + self.weights_path + '/v_embedding_ae_weights.h5')
