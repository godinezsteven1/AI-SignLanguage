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
import json

envPath = os.path.join(os.path.dirname(__file__), '..', 'secureClientInfo.env') #your file name here 
load_dotenv(envPath)

client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")
user_agent = os.environ.get("USER_AGENT")
postLimit = os.environ.get("POST_LIMIT")
postLimit = int(postLimit)
redditName = "AskReddit"
#redditName = "neu"
print("CLIENT_ID =", client_id)
print("CLIENT_SECRET =", client_secret)
print("USER_AGENT =", user_agent)
print("POST_LIMIT =", postLimit)


#create reddit object 
def redditInit(client_id, client_secret, user_agent):
    reddit = praw.Reddit(
        client_id = client_id,
        client_secret = client_secret,
        user_agent = user_agent)
    return reddit

#scrape mass reddit for probabilities
def redditScraper(reddit,subreddit_name = redditName):
    text = [] 
    subreddit = reddit.subreddit(subreddit_name)
    print("scraping reddit: ", subreddit_name)

    #for each post skip over NSFW content,collect the text 
    for sub in subreddit.top(limit=postLimit):
        #print("reached loop")
        if sub.over_18:
            #print("sub.over_18")
            continue 
        if sub.selftext:
            #print("reached sub.selftext")
            text.append(sub.selftext) #look at doc for this how is it boolean and string at same time
        
        #loop in comments be careful w deleted/removed comments 
        sub.comments.replace_more(limit=0)
        for comment in sub.comments: #comments as list
            if comment is None or comment.body is None: # skip null pointers
                continue
            #if comment.over_18: #not sure if this func also works for comment bodies thats why its separated 
            #    continue  #does not exist 
            text.append(comment.body)
            print(comment.body)
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










##########################
# FOLLOWING ARE UPDATED TO ACCOUNT FOR APPENDING DATA and increasing dataset size 
##########################

#same as startPorbHMM except doesn ot calc prob, returns counter 
def countStart(textFile):
    startCounter = Counter()
    for words in textFile:
        token = cleanAndSplit(words)
        if token:
            startCounter[token[0]] += 1
    return startCounter

# split into two funcs 
# Prob(interpreted as dictionary of dictionary 
#def transProbHMM(textFile):
#    freq = defaultdict(Counter)
#    for words in textFile:
#        token = cleanAndSplit(words)
#        for i in range(len(token) - 1): # 0 based
#            w1, w2 = token[i], token[i+1] # w1 =i w2 = i+1
#            freq[w1][w2] += 1 #increment to next entry
#    transitionProb = {}
#    for w1, counter in freq.items():
#        total = sum(counter.values()) # or should it be freq.values
#        transitionProb[w1] = {w2: count / total for w2, count in counter.items()} #chat helped initialize this one line
#    return transitionProb

def countTrans(textFile):
    transCount = defaultdict(Counter)
    for words in textFile:
        token = cleanAndSplit(words)
        for i in range(len(token) - 1):
            w1, w2 = token[i], token[i + 1]
            transCount[w1][w2] += 1
    return transCount



# emissions 
#def emissionProbHMM(textFile):
#    frequency = frequencyMaker(textFile)
#    emissionProb = {}
#    for word in frequency:
#        emissionProb[word] = {word: 1.0} # why doesnt this work as a int? # leave as float. 
#    return emissionProb
def countEmission(textFile):
    allWords = []
    for words in textFile:
        word = cleanAndSplit(words)
        allWords.extend(word)
    return Counter(allWords)

def mergeCount(old, new):
        return old + new

def mergeCounters(old, new):
    merged = defaultdict(Counter, old)
    for key, counter in new.items():
        merged[key].update(counter)
    return merged

def calcProbabilities(counter):
    total = sum(counter.values())
    return {word: count / total for word, count in counter.items()} #from og startProbHMM

def calcProbNested(nestCounter):
    prob = {}
    for key, counter in nestCounter.items():
        total = sum(counter.values())
        prob[key] = {word: count / total for word, count in counter.items()} #from og probability funcs
    return prob

def loadJSONFile(name):
    #never pass one that does not exist but just in case
    if os.path.exists(name):
        with open(name, "r") as f: #read 
            return json.load(f)
    else:
        print(f"file does not exist: {name}")
        return None
    
def saveJSONFile(data, name):
    with open(name, "w") as f: #write
        json.dump(data, f)

def main():

    reddit = redditInit(client_id, client_secret, user_agent)
    text = redditScraper(reddit, subreddit_name=redditName)
    print(f"confirmation, scanning {redditName}")

    # new counts 
    newStartCounts = countStart(text)          
    newTransCounts = countTrans(text)    
    newEmissionCounts = countEmission(text)

    # load existing count if they exist
    loadStartCount = loadJSONFile("startCount.json")
    loadTransCount = loadJSONFile("transitionCount.json")
    loadEmissionCount = loadJSONFile("emissionCount.json")

    # if no exist intitialize 
    if loadStartCount is None:
        loadStartCount = {}
    if loadTransCount is None:
        loadTransCount = {}
    if loadEmissionCount is None:
        loadEmissionCount = {}

    # merge old with new aka empty and non empty, or non and non
    mergedStartCount = mergeCount(Counter(loadStartCount), newStartCounts)
    mergedTransCount = mergeCounters(loadTransCount, newTransCounts)
    mergedEmissionCount = mergeCount(Counter(loadEmissionCount), newEmissionCounts)

    # update probabilities
    updatedStartProb = calcProbabilities(mergedStartCount)
    updatedTransProb = calcProbNested(mergedTransCount)
    updatedEmissionProb = {"start": calcProbabilities(mergedEmissionCount)}

    # file update with merge
    saveJSONFile(dict(mergedStartCount), "startCount.json")
    saveJSONFile(mergedTransCount, "transitionCount.json")
    saveJSONFile(dict(mergedEmissionCount), "emissionCount.json")

    #file update with probs
    saveJSONFile(updatedStartProb, "startProb.json")
    saveJSONFile(updatedTransProb, "transitionProb.json")
    saveJSONFile(updatedEmissionProb, "emissionProb.json")
    

    return updatedStartProb, updatedTransProb, updatedEmissionProb

    return startProb, transitionProb, emissionProb

#scripting use 
if __name__ == "__main__":
    sProb, tProb, eProb = main()
    print("start prob")
    #print(sProb)
    print("trans prob")
    #print(tProb)
    print("emission prob")
    #print(eProb)
    


