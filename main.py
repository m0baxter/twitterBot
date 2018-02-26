
import time
import sys
import random as rnd
import matplotlib
import matplotlib.pyplot as plt
from src.twitterBot import *

sleepTimes = [ 225, 450, 900, 1800, 3600, 7200, 14400 ]

batchSize = 1024 #1024
nh = 256 #should this be 280?
nl = 1 #2
nEpochs = 2400
matplotlib.rcParams.update({'font.size': 24, 'text.usetex': True})


def plotLosses( losses ):
    """Plots training loss as a fucntion of epoch."""

    fig = plt.figure(1, figsize = (18,10))
    plt.plot( range(1, len(losses) + 1), losses, "b-", linewidth = 3)
    plt.ylabel("$\mathrm{Training}$ $\mathrm{Loss}$")
    plt.xlabel("$\mathrm{Epoch}$")
    fig.savefig( "lossPlot.eps", format = 'eps', dpi = 20000, bbox_inches='tight' )

    return


if __name__ == "__main__":

    assert len(sys.argv) == 3, "Invalid command line arguments."

    if (sys.argv[1] == "train"):

        bot = TwitterBot( sys.argv[2], True, nHidden = nh, numLayers = nl )

        #Uncomment to continue training:
        #bot.load( "./weights/trained.hdf5" )

        losses = bot.trainBot( batchSize = batchSize, nEpochs = nEpochs )
        bot.save("./weights/trained.hdf5")

        plotLosses( losses )

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

        for _ in range(5):
            print "Sample tweet:\n   ", bot.genTweet( 1.0 ), "\n"

    else:
        print ( "Invalid argument, must be either 'tweet', 'train', or 'test'." )

