from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import Input
from keras.models import Model

def fcn(pretrained_weights=None, input_size=(360, 224, 3)):
	pool_size = (2, 2)
	inputs = Input(input_size)

	bn = BatchNormalization()(inputs)
	conv_1 = Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1')(bn)
	conv_2 = Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2')(conv_1)
	max_1 = MaxPooling2D(pool_size=pool_size)(conv_2)
	conv_3 = Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3')(max_1)
	dropout_1 = Dropout(0.2)(conv_3)
	conv_4 = Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4')(dropout_1)
	dropout_2 = Dropout(0.2)(conv_4)
	conv_5 = Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5')(dropout_2)
	dropout_3 = Dropout(0.2)(conv_5)
	max_2 = MaxPooling2D(pool_size=pool_size)(dropout_3)
	conv_6 = Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6')(max_2)
	dropout_4 = Dropout(0.2)(conv_6)
	conv_7 = Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7')(dropout_4)
	dropout_5 = Dropout(0.2)(conv_7)
	max_3 = MaxPooling2D(pool_size=pool_size)(dropout_5)
	upsample_1 = UpSampling2D(size=pool_size)(max_3)
	deconv_1 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1')(upsample_1)
	dropout_6 = Dropout(0.2)(deconv_1)
	deconv_2 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2')(dropout_6)
	dropout_7 = Dropout(0.2)(deconv_2)
	upsample_2 = UpSampling2D(size=pool_size)(dropout_7)
	deconv_3 = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3')(upsample_2)
	dropout_8 = Dropout(0.2)(deconv_3)
	deconv_4 = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4')(dropout_8)
	dropout_9 = Dropout(0.2)(deconv_4)
	deconv_5 = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5')(dropout_9)
	dropout_10 = Dropout(0.2)(deconv_5)
	upsample_3 = UpSampling2D(size=pool_size)(dropout_10)
	deconv_6 = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6')(upsample_3)
	final = Conv2DTranspose(3, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final')(deconv_6)

	model = Model(input = inputs, output = final)
	model.summary()
	model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['acc'])

	if pretrained_weights:
		model.load_weights(pretrained_weights)

	return model