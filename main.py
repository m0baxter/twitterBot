
import time
import sys
import random as rnd
import matplotlib
import matplotlib.pyplot as plt
from src.twitterBot import *

sleepTimes = [ 1800, 3600, 7200, 14400 ]

batchSize = 256
nh = 256
nl = 3
nEpochs = 3000
matplotlib.rcParams.update({'font.size': 24, 'text.usetex': True})


def plotLosses( losses ):
    """Plots training loss as a fucntion of epoch."""

    fig = plt.figure(1, figsize = (18,10))
    plt.plot( range(1, len(losses["loss"]) + 1), losses["loss"], "b-",
              linewidth = 3, label = "$\mathrm{training}$")
    plt.plot( range(1, len(losses["val_loss"]) + 1), losses["val_loss"], "g-",
              linewidth = 3, label = "$\mathrm{validation}$")
    plt.ylabel("$\mathrm{Loss}$")
    plt.xlabel("$\mathrm{Epoch}$")
    plt.legend( loc = "best" )
    fig.savefig( "lossPlot.eps", format = 'eps', dpi = 20000, bbox_inches='tight' )

    return


if __name__ == "__main__":

    assert len(sys.argv) == 3, "Invalid command line arguments."

    if (sys.argv[1] == "train"):

        bot = TwitterBot( sys.argv[2], True, nHidden = nh, numLayers = nl )

        print bot.model.summary()

        #Uncomment to continue training:
        #bot.load( "./weights/trained.hdf5" )

        losses = bot.trainBot( batchSize = batchSize, nEpochs = nEpochs,
                               savePath = "./weights/trained.hdf5" )

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
        print bot.model.summary()
        bot.load( "./weights/trained.hdf5" )

        for _ in range(5):
            print "\n\nSample tweet:\n   ", bot.genTweet( rnd.random() ), "\n"

    else:
        print ( "Invalid argument, must be either 'tweet', 'train', or 'test'." )

