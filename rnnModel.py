
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
sys.stderr = stderr
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

def genCodedText( model, nChars, phraseLen = 282 ):
    """Generates a phrase given of length phraseLen. Starts from a random character."""

    x = np.zeros( (1, phraseLen, nChars + 3) )
    x[0, 0, :][nChars] = 1 #make first character the start character.

    xi = nChars 

    phrase = [nChars ]

    for i in range(phraseLen):
        x[0, i, :][xi] = 1

        probDist = model.predict(x[:, :i+1, :])[0, i]

        xi = np.random.choice( range(nChars + 3), p = probDist.ravel())

        phrase.append( xi )

    return phrase

