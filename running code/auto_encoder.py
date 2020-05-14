import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout
from tensorflow.keras.models import Model
import os


class AutoEncoder:
    def __init__(self, input_set, weights_path, encoding_dim=3):
        self.encoding_dim = encoding_dim
        self.x = input_set
        self.input_shape = len(input_set[0])
        # different way
        # self.input_shape = input_set[0].shape[0]
        self.num_classes = 2  # binary classifier
        self.weights_path = weights_path

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
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        decoded3 = Dense(self.input_shape, activation='elu')(dropout2)
        reshape = Reshape((int(self.input_shape / 21), 21))(decoded3)
        decoded3 = Dense(21, activation='softmax')(reshape)
        reshape2 = Reshape(self.x[0].shape)(decoded3)
        model = Model(inputs, reshape2)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        self.autoencoder_model = model
        return model

    def _encoder_2(self):
        inputs = Input(shape=self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu', name='last_layer')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder_2 = model
        return model

    def fc(self, enco):
        fc1 = Dense(30, activation='tanh')(enco)
        fc2 = Dense(15, activation='tanh')(fc1)
        dropout1 = Dropout(0.1)(fc2)
        fc3 = Dense(10, activation='tanh')(dropout1)
        dropout2 = Dropout(0.1)(fc3)
        out = Dense(1, activation='sigmoid')(dropout2)
        return out

    def classifier(self):
        ec = self._encoder_2()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        model = Model(inputs, self.fc(ec_out))
        self.classifier_model = model
        return model

    def fit_autoencoder(self, train_x, batch_size=10, epochs=300):
        # self.autoencoder_model = multi_gpu_model(self.autoencoder_model, gpus=3)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.autoencoder_model.compile(optimizer=adam, loss='mse')
        log_dir = './log/'
        tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')
        results = self.autoencoder_model.fit(train_x, train_x, validation_split=0.2, verbose=2,
                                             epochs=epochs, batch_size=batch_size,
                                             callbacks=[tb_callback, es_callback])
        return results

    def fit_classifier(self, train_x, train_y, batch_size=10, epochs=300):
        # self.classifier_model = multi_gpu_model(self.classifier_model, gpus=3)
        adam = tensorflow.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        log_dir = './log/'
        tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')

        # # train only fc
        # for layer in self.encoder_2.layers:
        #     layer.trainable = False
        # self.classifier_model.compile(optimizer=adam, loss='binary_crossentropy',
        #                               metrics=['accuracy'])
        # results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2,
        #                           epochs=epochs, batch_size=batch_size,
        #                           callbacks=[tb_callback, es_callback])
        #
        # # train both encoder and fc
        # for layer in self.encoder_2.layers:
        #     layer.trainable = True
        self.classifier_model.compile(optimizer=adam, loss='binary_crossentropy',
                                      metrics=['acc'])
        results = self.classifier_model.fit(train_x, train_y, validation_split=0.2, verbose=2,
                                            epochs=epochs, batch_size=batch_size,
                                            callbacks=[tb_callback])
        return results

    def save_ae(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.encoder.save(r'./weights_' + self.weights_path + '/encoder_weights.h5')
        self.decoder.save(r'./weights_' + self.weights_path + '/decoder_weights.h5')
        self.autoencoder_model.save(r'./weights_' + self.weights_path + '/ae_weights.h5')

    def save_cl(self):
        if not os.path.exists(r'./weights_' + self.weights_path):
            os.mkdir(r'./weights_' + self.weights_path)
        self.classifier_model.save(r'./weights_' + self.weights_path + '/classifier_weights.h5')
