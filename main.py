
from prepData import *
from rnnModel import *


batchSize = 16
nh = 64
nl = 1


if __name__ == "__main__":

    X, c2i, i2c = genData( "trump.json" )

    nChars = len(c2i)

    model = genModel( nChars + 3, nHidden = nh, numLayers = nl )

    model.fit_generator( genBatches( X, nChars + 3, batchSize ),
                           steps_per_epoch = len(X)/batchSize,
                           epochs = 10 )

    coded = genCodedText( model, nChars, phraselen = 282 )

    print decodeString( coded, i2c, nChars, nChars + 1, nChars + 2 )

