# twitterBot
A twitter bot that will learn to tweet like a person.

The twitter bot has three modes of operation: `train`, `test`, and `tweet`. Which ever mode is chosen
the code may be run by calling

```
python main.py $mode $file
```

where file is the location of a `json` formatted file of tweet data.

It should be noted thateven without Twitter authentication keys training and testing are possible. This
is achieved by setting the the `online` argument of the `TwitterBot` constructor to `False`.

## Training
`main.py` contains sample parameters used to train a twitter bot on  tweet data from the [Trump Twitter
Archive](http://www.trumptwitterarchive.com/archive). Beyond changing all occurrences of `&amp;` to
`&` in tweet data no further preprocessing of the data took place. Sample weights obtained from training
are provided in `./weights`.

If desired one may change the parameters defining the RNN architecture. The most relevant parameters are
`nl` and `nh` which control the number and size of LSTM units used. It should be noted that retraining
is necessary to change run the bot with anything other than the default parameters.

## Testing
Running the code in test mode will generate five sample tweets. The level of randomness in may be
controlled using the parameter passed to the method `TwitterBot.genTweet()` (0.0 being completely
deterministic and 1.0 sampling the probability distribution at each time step).

## Tweeting
To run the twitter bot one must obtain authorization credentials. Once this is done create place them
in `./authorization/keys.auth` one string per line in the format:

```
Consumer Key (API Key)
Consumer Secret (API Secret)
Access Token
Access Token Secret
```

The application must be given write access for tweets to be automatically posted.

A sample of the twitter bot in action can be found [here](https://twitter.com/trumptron9000).

