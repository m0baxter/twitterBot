
import tweepy as twp
from prepData import *
from rnnModel import *
from keras.callbacks import EarlyStopping, ModelCheckpoint


class TwitterBot(object):

    def __init__( self, dataPath, trainable, online = True,
                  nHidden = 140, numLayers = 1, savePath = "weights.hdf5" ):

        X, c2i, i2c = genData( dataPath )
        self.data = None
        self.trainable = trainable

        if ( trainable ):
            self.data = X

        self.c2i = c2i
        self.i2c = i2c
        self.nChars = len(c2i)
        self.model = genModel( self.nChars + 3, nHidden = nHidden, numLayers = numLayers )

        self.api = None

        if ( online == True ):
            consumerKey, consumerSecret, userToken, userSecret = self.__readTokens()
            auth = twp.OAuthHandler(consumerKey, consumerSecret)
            auth.set_access_token( userToken, userSecret)
            self.api = twp.API(auth)

    def __readTokens( self ):
        """Reads the authentication file."""

        try:
            with open( "./authorization/keys.auth" ) as readFile:
                text = readFile.read()
                consumerKey, consumerSecret, userToken, userSecret = text.split("\n")[:4]

                return consumerKey, consumerSecret, userToken, userSecret

        except ( ValueError, IOError ):
            print ("Could not find valid twitter authentication tokens.")

    def load( self, path ):
        """Loads a pretrained model from path."""

        self.model.load_weights( path )

    def save( self, path ):
        """Saves model weights."""

        self.model.save_weights( path )

    def trainBot( self, batchSize = 64, nEpochs = 100, savePath = "weights.hdf5" ):

        losses = None

        if ( self.trainable ):
            train, val = splitData( self.data, 0.2 )

            earlyStoper  = EarlyStopping( patience = 10 )
            checkPointer = ModelCheckpoint( filepath = savePath, verbose = 1, save_best_only=True )

            losses = self.model.fit_generator( genBatches( train, self.nChars + 3, batchSize ),
                                    steps_per_epoch = len(train)/batchSize, epochs = nEpochs,
                                    validation_data = genBatches( val, self.nChars + 3, batchSize ),
                                    validation_steps = len(val)/batchSize,
                                    callbacks = [earlyStoper, checkPointer] )

        else:
            print "TwitterBot object was not created to be trainable."

        return losses.history

    def genTweet( self, rnd = 1.0 ):

        coded = genCodedText( self.model, self.nChars, phraseLen = 282, rndLevel = rnd )

        return decodeString( coded, self.i2c, self.nChars, self.nChars + 1, self.nChars + 2 )

    def sendTweet( self, line ):

        self.api.update_status( line )

