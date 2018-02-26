
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed
from keras.optimizers import Adam
#from keras.callbacks import EarlyStopping
import random as rnd
import numpy as np


def genModel( nChars, nHidden, numLayers = 1 ):
    """Generates the RNN model with nChars characters and numLayers hidden units with
       dimension nHidden."""

    model = Sequential()
    model.add( LSTM( nHidden, input_shape = (None, nChars), return_sequences = True ) )

    for _ in range( numLayers - 1 ):
        model.add( LSTM( nHidden, return_sequences = True) )

    model.add( Dropout(0.4) )
    model.add( TimeDistributed( Dense(nChars) ) )
    model.add( Activation('softmax') )

    opt = Adam( amsgrad = True )

    model.compile( loss = "categorical_crossentropy", optimizer = opt )

    return model

def genCodedText( model, nChars, phraseLen = 282, rndLevel = 1.0 ):
    """Generates a phrase of length phraseLen. Starts from a random character.
       rndLevel allows one to tune how deterministic the sampling will be."""

    x = np.zeros( (1, phraseLen, nChars + 3) )
    x[0, 0, :][nChars] = 1 #make first character the start character.

    xi = nChars

    phrase = [nChars ]

    for i in range(phraseLen):
        x[0, i, :][xi] = 1

        probDist = model.predict(x[:, :i+1, :])[0, i]

        if ( rnd.random() < rndLevel):
            xi = np.random.choice( range(nChars + 3), p = probDist.ravel())

        else:
            xi = np.argmax( probDist )

        phrase.append( xi )

    return phrase

