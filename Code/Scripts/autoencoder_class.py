import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import keras.backend as K
from keras.layers import Lambda
import numpy as np
from normalizer_class import Normalizer

''' Класс автоэнкодеров '''
class AutoencoderClass():
  def __init__(self, func, input_dim : int, encoding_dim : int, activations : list, enc_type : str, normalizer : Normalizer):
    self.func = func                 # Функция обучения
    self.batch = 0                   # Размр батча
    self.input_dim = input_dim       # Размерность входного представления
    self.encoding_dim = encoding_dim # Размерность кодированного представления
    self.activations = activations   # Функции активации
    self.enc_type = enc_type         # Тип автоэнкодера
    self.aes_types = {'dense': self.__create_dense_ae,
                      'deep':  self.__create_deep_dense_ae,
                      'conv':  self.__create_deep_conv_ae,
                      'vae':   self.__create_vae}
    self.normalizer = normalizer
    try:
      # Сборка моделей
      self.encoder, self.decoder, self.autoencoder = self.aes_types[self.enc_type]()
      if self.aes_types != 'vae':
        self.autoencoder.compile(optimizer = 'adam', loss = self.custom_loss, metrics=['accuracy'])
      else:
        self.autoencoder.compile(optimizer = 'adam', loss = self.vae_loss, metrics=['accuracy'])
    except KeyError as e:
      raise ValueError('Undefined unit: {}'.format(e.args[0]))

  # Обучение модели
  def fit(self, train_data, test_data, epochs : int, batch_size : int, shuffle : bool):
    self.batch = batch_size
    if self.enc_type != 'conv':
      self.autoencoder.fit(train_data, train_data,
                           epochs=epochs,
                           batch_size=self.batch,
                           shuffle=shuffle,
                           validation_data=(test_data, test_data))
    else:
      grid_train = []
      grid_test = []
      for i in range(len(train_data)):
        xx, yy = np.meshgrid(train_data[i], train_data[i])
        grid_train.append(xx)

      for i in range(len(test_data)):
        xx, yy = np.meshgrid(test_data[i], test_data[i])
        grid_test.append(xx)
      
      self.autoencoder.fit(grid_train, grid_train,
                           epochs=epochs,
                           batch_size=self.batch,
                           shuffle=shuffle,
                           validation_data=(grid_test, grid_test))

  # Предсказание результата
  def predict(self, x_vector):
    if self.enc_type != 'conv':
      return self.autoencoder.predict(x_vector)
    else:
      return self.autoencoder.predict(x_vector)[0]

  # Тип автоэнкодера
  def get_aec_type(self):
    return self.enc_type

  # Возвращает собранные модели
  def get_models(self):
    return self.autoencoder, self.encoder, self.decoder

  # Loss функция
  @tf.autograph.experimental.do_not_convert
  def custom_loss(self, x_true, x_pred):
    return K.mean(K.abs(self.func(self.normalizer.renormalize(x_pred)[0]) - self.func(self.normalizer.renormalize(x_true)[0])))

  # Loss функция для вариационного автоэнкодера
  @tf.autograph.experimental.do_not_convert
  def vae_loss(self, x_true, x_pred):
    x_true = K.reshape(x_true, shape=(self.batch, self.input_dim))
    x_pred = K.reshape(x_pred, shape=(self.batch, self.input_dim))
    loss = self.custom_loss(x_true, x_pred)
    kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var))
    return loss + kl_loss

  ''' Сжимающий автоэнкодер '''
  def __create_dense_ae(self):
    # Энкодер
    input_data = Input(shape=(self.input_dim))
    encoded = Dense(self.encoding_dim, activation = self.activations[0])(input_data)
    
    # Декодер
    input_encoded = Input(shape = (self.encoding_dim))
    decoded = Dense(self.input_dim, activation = self.activations[1])(input_encoded)

    # Модели
    encoder = Model(input_data, encoded, name = "encoder")
    decoder = Model(input_encoded, decoded, name = "decoder")
    autoencoder = Model(input_data, decoder(encoder(input_data)), name = "autoencoder")
    return encoder, decoder, autoencoder

  ''' Глубокий автоэнкодер '''
  def __create_deep_dense_ae(self):
    # Энкодер
    input_data = Input(shape=(self.input_dim))
    x = Dense(self.encoding_dim*2, activation='relu')(input_data)
    encoded = Dense(self.encoding_dim, activation='linear')(x)
    
    # Декодер
    input_encoded = Input(shape=(self.encoding_dim,))
    x = Dense(self.encoding_dim*2, activation='relu')(input_encoded)
    decoded = Dense(self.input_dim, activation='sigmoid')(x)
    
    # Модели
    encoder = Model(input_data, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_data, decoder(encoder(input_data)), name="autoencoder")
    return encoder, decoder, autoencoder

  ''' Сверточный автоэнкодер '''
  def __create_deep_conv_ae(self):
    # Энкодер
    input_data = Input(shape=(self.input_dim, self.input_dim, 1))
    x = Conv2D(25, (2, 2), activation='relu', padding='same')(input_data)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (2, 2), activation='relu', padding='same')(x)

    # На этом моменте представление  (7, 7, 1) т.е. 49-размерное
    
    # Декодер
    input_encoded = Input(shape=(7, 7, 1))
    #x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_encoded)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2D(25, (2, 2), activation='relu', padding='same')(input_encoded)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

    # Модели
    encoder = Model(input_data, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_data, decoder(encoder(input_data)), name="autoencoder")
    return encoder, decoder, autoencoder

  ''' Вариационный автоэнкодер                         '''
  ''' Работает на основе девергенции Кульбака-Лейблера '''
  ''' Идея: переход данных скрытого слоя к нормальному распределению'''
  ''' Статья: https://habr.com/ru/post/484756/ '''
  ''' Видео:  https://youtu.be/ebI3JLAcWqQ '''
  def __create_vae(self):
    hidden_dim = 2

    input_data = Input(shape=(self.input_dim))
    x = Dense(self.encoding_dim, activation='relu')(input_data)
    
    self.z_mean = Dense(self.encoding_dim)(x)    # Мат ожидание
    self.z_log_var = Dense(self.encoding_dim)(x) # Логарифм дисперсии
    
    # Нормальное распределение N(0, 1)
    def noiser(args):
      self.z_mean, self.z_log_var = args
      N = K.random_normal(shape=(self.batch, self.encoding_dim), mean=0., stddev=1.0)
      return K.exp(self.z_log_var / 2) * N + self.z_mean
    
    # Преобразование данных в нормальное распределения
    h = Lambda(noiser, output_shape=(self.encoding_dim,))([self.z_mean, self.z_log_var])
    
    input_encoded = Input(shape=(self.encoding_dim,))
    d = Dense(self.encoding_dim, activation='relu')(input_encoded)
    decoded = Dense(self.input_dim, activation='sigmoid')(d)
    
    encoder = Model(input_data, h, name='encoder')
    decoder = Model(input_encoded, decoded, name='decoder')
    vae = Model(input_data, decoder(encoder(input_data)), name="vae")
    return encoder, decoder, vae
