
import json
import pandas as pd
import numpy as np
from keras.utils import to_categorical

def removeURL( s ):
    """Removes all URLs from the string s."""

    keep = []

    for w in s.split():
        if ( not "http" in w ):
            keep.append(w)

    return " ".join(keep)

def charIntMaps( chars ):
    """Generates encoding/decoding dictionaries"""

    charToInt = { c : i for i, c in enumerate( sorted(chars) ) }
    intToChar = { i : c for i, c in enumerate( sorted(chars) ) }

    return ( charToInt, intToChar )

def encodeString( s, char2Int, start, end, pad, maxLen = 280 ):
    """encodes a string into integers."""

    return [start] + map( lambda x : char2Int[x], s) + [end] + (maxLen - len(s))*[pad]

def decodeString( enc, int2Char, start, end, pad ):
    """Decodes a string from integers to characters."""

    result = ""

    for i in enc:
        if ( not i in [start, end, pad] ):
            result += int2Char[i]

    return result

def getChars( data ):
    """Generates a set of all characters in the data set."""

    chars = set([])

    for i in range( len(data) ):
        chars.update( set( data["text"][i] ) )

    return chars

def genData( path ):
    """Reads data from path and generates training data and the encoding/decoding maps."""

    data = pd.read_json( path )
    data["text"] = data["text"].apply(removeURL)

    c2i, i2c = charIntMaps( getChars( data ) )
    nc = len(c2i)

    X = []

    for i in range(len(data)):
        X.append( encodeString( data['text'][i], c2i,
                                start = nc, end = nc + 1, pad = nc + 2,
                                maxLen = 280 ) )

    X = np.array(X)

    return X, c2i, i2c

def splitData( X, valFrac ):
    """Splits the data into training and validation sets. valFrac is the fraction of total
       that makes up the validation set."""
    
    n = len(X)
    inds = np.random.permutation( len(X) )
    trainInds, valInds = inds[ : -int(n*valFrac) ], inds[ -int(n*valFrac) : ]
    
    return ( X[ trainInds ], X[ valInds ] )

def genBatches( X, numClasses, batchSize = 16 ):
    """Generator of one-hot encoded batches for training."""

    while (True):
        inds = np.random.permutation( len(X) )

        for start in range(0, len(X) - 1, batchSize):
            temp = to_categorical( X[ inds[start : start + batchSize] ] )
            batchX = temp[ : ,   : -1, : ]
            batchY = temp[ : , 1 :   , : ]

            yield batchX, batchY

