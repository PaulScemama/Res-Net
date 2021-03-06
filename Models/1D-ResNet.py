from tensorflow import keras

class RN1D_Classifier:

    def __init__(self, input_shape, n_classes, kernel_size = 10):
        self.input_shape = input_shape
        self.classes     = n_classes
        self.kernel_size = kernel_size
    
    def rn_block(self, input, n_filters, kernel_size, downsample = False):
        # ID is what we will add at the end of the block // stride is defaulted to 1
        ID     = input
        stride = 1

        # if we downsample, (when we want to increase the number of filters / downsize resolution)
        # we need to project the input with a 1x1 convolution in order for the addition at the end of the block to work
        # also, the stride for the convolution afterward needs to be 2 to downsize the resolution of the manipulated input
        if downsample:
            ID = keras.layers.Conv1D(filters = n_filters, kernel_size = 1, padding = 'same', strides = 2)(ID)
            stride = 2
        
        conv1  = keras.layers.Conv1D(filters = n_filters, kernel_size = kernel_size, padding = 'same', strides = stride, kernel_initializer = 'he_normal')(input)
        batch1 = keras.layers.BatchNormalization()(conv1)
        relu1  = keras.layers.Activation('relu')(batch1)

        conv2  = keras.layers.Conv1D(filters = n_filters, kernel_size = kernel_size, padding = 'same', strides = 1, kernel_initializer = 'he_normal')(relu1)
        batch2 = keras.layers.BatchNormalization()(conv2)
        
        # we don't apply relu until after adding the ID (input) to the output of the residual block
        pre_output = keras.layers.add([batch2, ID])
        output     = keras.layers.Activation('relu')(pre_output)

        return output

    
    def model(self):
        n_feature_maps = 32

        input_layer = keras.layers.Input(self.input_shape)

        ID = input_layer

        # Layer 0
        conv1  = keras.layers.Conv1D(filters = n_feature_maps, kernel_size = self.kernel_size, padding = 'same')(input_layer)
        batch =  keras.layers.BatchNormalization()(conv1)
        relu   = keras.layers.Activation('relu')(batch)


        ## SECTION 1
        output1 = self.rn_block(relu, n_feature_maps, kernel_size = 10)
        output2 = self.rn_block(output1, n_feature_maps, kernel_size = 10)
        
        # Section 2
        # increase number of feature maps by two-fold
        n_feature_maps *= 2
        output3 = self.rn_block(output2, n_feature_maps, kernel_size = 5, downsample = True)
        output4 = self.rn_block(output3, n_feature_maps, kernel_size = 10)
        output5 = self.rn_block(output4, n_feature_maps, kernel_size = 15)
        output6 = self.rn_block(output5, n_feature_maps, kernel_size = 20)


        x           = keras.layers.AveragePooling1D(pool_size = 5, padding = 'valid')(output6)
        x           = keras.layers.Flatten()(x)
        x           = keras.layers.Dense(self.classes, activation = 'softmax')(x)


        model = keras.Model(inputs = [input_layer], outputs = [x])
        
        return model


