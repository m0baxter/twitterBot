
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
import numpy as np


def genModel( nChars, nHidden, numLayers = 1 ):
    """Generates the RNN model with nChars characters and numLayers hidden units with
       dimension nHidden."""

    model = Sequential()
    model.add( LSTM( nHidden, input_shape = (None, nChars), return_sequences = True ) )

    for i in range( numLayers - 1 ):
        model.add( LSTM( nHidden, return_sequences = True) )

    model.add( TimeDistributed( Dense(nChars) ) )
    model.add( Activation('softmax') )

    model.compile( loss = "categorical_crossentropy", optimizer = "adam" )

    return model

def genCodedText( model, nChars, phraselen = 282 ):
    """Generates a phrase given of length phraseLen. Starts from a random character."""

    x = np.zeros( (1, phraseLen, nChars + 3) )
    x[1, 0, :][nChars] = 1 #make first character the start character.

    xi = [ np.random.randint(nChars) ]

    phrase = [start, xi]

    for i in range(1, phraseLen):
        X[0, i, :][xi[-1]] = 1

        xi = np.argmax( model.predict(X[:, :i+1, :])[0], 1 )

        phrase.append( xi[-1] )

    return phrase

