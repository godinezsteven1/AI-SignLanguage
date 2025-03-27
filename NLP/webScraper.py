#purpose of this file is to web srape reddit in order to get 
# - transition probabilities
# - emission probabilities 
# - start probabilities 
# for the viterbi algorithm in HMM file

import praw
import re
from collections import Counter, defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")
user_agent = os.environ.get("USER_AGENT")
postLimit = os.environ.get("POST_LIMIT")


#create reddit object 
def redditInit(client_id, client_secret, user_agent):
    reddit = praw.Reddit(
        client_id = client_id,
        client_secret = client_secret,
        user_agent = user_agent)
    return reddit

#scrape mass reddit for probabilities
def redditScraper(reddit,subreddit_name = "AskReddit", postLimit):
    text = [] 
    subreddit = reddit.subreddit(subreddit_name)

    #for each post skip over NSFW content,collect the text 
    for sub in subreddit.best(limit=postLimit):
        if sub.over_18:
            continue 
        if sub.selftext:
            text.append(sub.selftext) #look at doc for this how is it boolean and string at same time
        
        #loop in comments be careful w deleted/removed comments 
        sub.comments.replace_more(limit=None)
        for comment in sub.comments.list(): #comments as list
            if comment is None or comment.body is None: # skip null pointers
                continue
            if comment.over_18: #not sure if this func also works for comment bodies thats why its separated 
                continue 
            text.append(comment.body)
    return text

#clean and split text
def cleanAndSplit(text):
    text = text.lower()
    # only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    token = text.split()
    return token

#count and set frequency
def frequencyMaker(text): 
    frequency = Counter()

    for sub in text:
        token = cleanAndSplit(sub)
        frequency.update(token)
    return frequency

#dict of word probabilities
def probabilities(frequency):
    total = sum(frequency.values())
    wordProb = {word: count / total for word, count in frequency.items()} #chat helped initialize this one liner
    return wordProb

#get start probabilities
def startProbHMM(textFile):
    startCounter = Counter()
    for words in textFile:
        token = cleanAndSplit(words)
        if token:
            startCounter[token[0]] += 1 #increment count 
    totalCounter = sum(startCounter.values()) # should be totla or some 
    startProb = {word: count / totalCounter for word, count in startCounter.items()} # chat helped initialize this one liner 
    return startProb


# Prob(word1 | word2) interpreted as dictionary of dictionary 
def transProbHMM(textFile):
    freq = defaultdict(Counter)
    for words in textFile:
        token = cleanAndSplit(words)
        for i in range(len(token) - 1): # 0 based
            w1, w2 = token[i], token[i+1] # w1 =i w2 = i+1
            freq[w1][w2] += 1 #increment to next entry
    transitionProb = {}
    for w1, counter in freq.items():
        total = sum(counter.values()) # or should it be freq.values
        transitionProb[w1] = {w2: count / total for w2, count in counter.items()} #chat helped initialize this one line
    return transitionProb



# emissions 
def emissionProbHmm(textFile):
    frequency = frequencyMaker(textFile)
    emissionProb = {}
    for word in frequency:
        emissionProb[word] = {word: 1.0} # why doesnt this work as a int? # leave as float. 
    return emissionProb
    


def main():

    reddit = redditInit(client_id, client_secret, user_agent)
    text = redditScraper(reddit, subreddit_name="AskReddit", postLimit)

    startProb = startProbHMM(text)
    transitionProb = transProbHMM(text)
    emissionProb = emissionProbHMM(text)


    # Return them (no printing, as requested)
    return startProb, transitionProb, emissionProb

#scripting use 
if __name__ == "__main__":
    sProb, tProb, eProb = main()
    


