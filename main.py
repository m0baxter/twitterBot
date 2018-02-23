
import time
import sys
import random as rnd
from src.twitterBot import *

sleepTimes = [ 225, 450, 900, 1800, 3600, 7200, 14400 ]

batchSize = 128
nh = 256
nl = 2
nEpochs = 200

if __name__ == "__main__":

    assert len(sys.argv) == 3, "Invalid command line arguments."

    if (sys.argv[1] == "train"):

        bot = TwitterBot( sys.argv[2], True, nHidden = nh, numLayers = nl )
        bot.trainBot( batchSize = batchSize, nEpochs = nEpochs )
        bot.save("./weights/trained.hdf5")

    elif (sys.argv[1] == "tweet"):

        bot = TwitterBot( sys.argv[2], False, nHidden = nh, numLayers = nl )
        bot.load( "./weights/trained.hdf5" )

        while (True):

            tweet = bot.genTweet()
            bot.sendTweet( tweet )

            print "Posted tweet:\n   ", tweet

            time.sleep( rnd.choice(sleepTimes) )

    elif (sys.argv[1] == "test"):

        bot = TwitterBot( sys.argv[2], False, nHidden = nh, numLayers = nl )
        bot.load( "./weights/trained.hdf5" )
        
        print bot.model.summary()

        for i in range(5):
            print "Sample tweet:\n   ", bot.genTweet()


    else:
        print ( "Invalid argument, must be either 'tweet', 'train', or 'test'." )

