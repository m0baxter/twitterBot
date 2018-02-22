
import tweepy as twp
from prepData import *
from rnnModel import *


class TwitterBot(object):
    
    def __init__(self, dataPath, trainable, nHidden = 140, numLayers = 1):
        
        X, c2i, i2c = genData( dataPath )
        self.data = None
        self.trainable = trainable
        
        if (trainable):
            self.data = X
            
        self.c2i = c2i
        self.i2c = i2c
        self.nChars = len(c2i)
        self.model = genModel( self.nChars + 3, nHidden = nHidden, numLayers = numLayers )
        
        consumerKey, consumerSecret, userToken, userSecret = self.__readTokens()
        auth = twp.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token( userToken, userSecret)
        self.api = twp.API(auth)
        
    def __readTokens(self):
        """Reads the authentication file."""

        try:
            with open("./authorization/keys.auth") as readFile:
                text = readFile.read()
                consumerKey, consumerSecret, userToken, userSecret = text.split("\n")[:4]

                return consumerKey, consumerSecret, userToken, userSecret

        except (ValueError, IOError):
            print ("Could not find valid twitter authentication tokens.")

                            
    def load(self, path):
        """Loads a pretrained model from path."""
        
        self.model.load_weights( path )

    def save(self, path):
        """Saves model weights."""

        self.model.save_weights( path )

    def trainBot(self, batchSize = 64, nEpochs = 100):

        if (self.trainable):
            self.model.fit_generator( genBatches( self.data, self.nChars + 3, batchSize ),
                                         steps_per_epoch = len(self.data)/batchSize, epochs = nEpochs )
        else:
            print "TwitterBot object was not created to be trainable."

    def genTweet(self):

        coded = genCodedText( self.model, self.nChars, phraseLen = 282 )

        return decodeString( coded, self.i2c, self.nChars, self.nChars + 1, self.nChars + 2 )

    def sendTweet( self, line ):

        self.api.update_status(line)


if __name__ == "__main__":

    tb = TwitterBot( "trump.json", False)

    tb.tweet("Test...\nTest...")

