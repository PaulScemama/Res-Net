from tensorflow import keras

class RN2D_Classifier:

    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.classes     = n_classes
    
    def cbr_layer(self, filters, kernel_size, stride = 1, activ = True):
        cbr = keras.models.Sequential()
        cbr.add(keras.layers.Conv2D(filters, kernel_size, padding = 'same', strides = stride))
        cbr.add(keras.layers.BatchNormalization())
        if activ:
            cbr.add(keras.layers.Activation('relu'))
        return cbr

    def brc_layer(self, filters, kernel_size, stride = 1):
        brc = keras.models.Sequential()
        brc.add(keras.layers.BatchNormalization())
        brc.add(keras.layers.Activation('relu'))
        brc.add(keras.layers.Conv2D(filters, kernel_size, padding = 'same', strides = stride))
        return brc
    
    def crb_layer(self, filters, kernel_size, stride = 1):
        crb = keras.models.Sequential()
        crb.add(keras.layers.Conv2D(filters, kernel_size, padding = 'same', strides = stride))
        crb.add(keras.layers.Activation('relu'))
        return crb

    def residual_block(self, input, filters, kernel_size, downsample = False, batchNormType = 'cbr'):
        if downsample:
            ID = keras.layers.Conv2D(filters = filters, kernel_size = 1, padding = 'same', strides = 2)(input)
            stride = 2
        else:
            ID = input
            stride = 1


        if batchNormType == 'cbr':
            layer1 = self.cbr_layer(filters, kernel_size, stride = stride)(input)
            layer2 = self.cbr_layer(filters, kernel_size, activ = False)(layer1)
            add    = keras.layers.add([layer2, ID])
            activ  = keras.layers.Activation('relu')(add)

            return activ
        
        elif batchNormType == 'brc':
            layer1 = self.brc_layer(filters, kernel_size, stride = stride)(input)
            layer2 = self.brc_layer(filters, kernel_size, activ = False)(layer1)
            add    = keras.layers.add([layer2, ID])
            activ  = keras.layers.Activation('relu')(add)

            return activ
        
        else:
            layer1 = self.crb_layer(filters, kernel_size, stride = stride)(input)
            layer2 = self.crb_layer(filters, kernel_size)(layer1)
            add    = keras.layers.add([layer2, ID])
            activ  = keras.layers.Activation('relu')(add)

            return activ

    
    def build_model(self, num_feature_maps, kernel_size, regularizer = None, batchNorm = True, batchNormtype = 'cbr'):
    
        n_feature_maps = num_feature_maps

        input_layer = keras.layers.Input(self.input_shape)

        ID = input_layer

        # Section 0
        conv   = keras.layers.Conv2D(filters = n_feature_maps, kernel_size = kernel_size, padding = 'same')(input_layer)
        batch  = keras.layers.BatchNormalization()(conv)
        relu   = keras.layers.Activation('relu')(batch)

        # Section 1
        block1 = self.residual_block(relu, num_feature_maps, kernel_size, batchNormType= batchNormtype)
        block2 = self.residual_block(block1, num_feature_maps, kernel_size, batchNormType= batchNormtype)

        # Section 2
        num_feature_maps *= 2
        block3 = self.residual_block(block2, num_feature_maps, kernel_size, downsample = True, batchNormType= batchNormtype)
        block4 = self.residual_block(block3, num_feature_maps, kernel_size, batchNormType= batchNormtype)

        # Section 3
        num_feature_maps *= 2
        block5 = self.residual_block(block4, num_feature_maps, kernel_size, downsample = True, batchNormType = batchNormtype)
        block6 = self.residual_block(block5, num_feature_maps, kernel_size, batchNormType=batchNormtype)

        x           = keras.layers.GlobalAveragePooling2D()(block6)
        x           = keras.layers.Flatten()(x)
        x           = keras.layers.Dense(self.classes, activation = 'softmax')(x)

        model = keras.Model(inputs = [input_layer], outputs = [x])
        return model







