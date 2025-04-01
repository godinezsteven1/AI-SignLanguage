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
commonStateLimit = 500
#postLimit = 2 #for debug 
#commonStateLimit = 10 # for debug

#redditName = "AskReddit"
#redditName = "neu"
#redditName = "todayilearned"
#redditName = "AmItheAsshole"
#redditName = "CasualConversation"
redditName = "confession"
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
    # only letters and spaces chat wrote line under this, forgot to add to Original commit (comment)
    text = re.sub(r'[^a-z\s]', '', text)
    token = text.split()
    return token


########
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
#########









##########################
# FOLLOWING ARE UPDATED TO ACCOUNT FOR APPENDING DATA and increasing dataset size 
##########################

#same as startPorbHMM except doesn ot calc prob, returns counter 
def countStart(textFile):
    startCounter = Counter()
    for sentence in textFile:
        tokens = cleanAndSplit(sentence)
        if tokens:
            startCounter[tokens[0]] += 1
    return startCounter

#word emissions set to 1 
def buildEmissionIdentity(wordSet):
    return {word: {word: 1.0} for word in wordSet}

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
            # increment the count of transitions from the current state to the next state
            transCount[token[i]][token[i + 1]] += 1 #chat helped initialize this 
    return transCount



# emissions 
#def emissionProbHMM(textFile):
#    frequency = frequencyMaker(textFile)
#    emissionProb = {}
#    for word in frequency:
#        emissionProb[word] = {word: 1.0} # why doesnt this work as a int? # leave as float. 
#    return emissionProb
def countEmission(textFile, wordToState):
    emissionCount = defaultdict(Counter)
    for words in textFile:
        token = cleanAndSplit(words)
        for word in token:
            state = wordToState.get(word, "rare")
            emissionCount[state][word] += 1
    return emissionCount


##########
def mergeCount(old, new):
        return old + new

def mergeCounters(old, new):
    merged = defaultdict(Counter, old)
    for key, counter in new.items():
        merged[key].update(counter)
    return merged
#############


def calcProbabilities(counter):
    total = sum(counter.values())
    return {word: count / total for word, count in counter.items()} #from og startProbHMM


def calcProbNested(nestCounter):
    return {state: calcProbabilities(counter) for state, counter in nestCounter.items()} #chat streamlined previous code for nested d


def loadJSONFile(name):
    #never pass one that does not exist but just in case
    if os.path.exists(name):
        with open(name, "r") as f: #read 
            return json.load(f)
    else:
        print(f"file does not exist: {name}")
        return None


# gona stick away from the mergers for now
# make file and indent this time
def saveJSONFile(data, name):
    with open(name, "w") as f: #write
        json.dump(data, f, indent=2)


#pipeline 
def main():
    reddit = redditInit(client_id, client_secret, user_agent)
    text = redditScraper(reddit, subreddit_name=redditName)

    newStart = countStart(text)
    newTrans = countTrans(text)

    # word stae map 
    freqCounter = frequencyMaker(text)
    # chat helped with this one liner 
    mostCommonStates = set([w for w, _ in freqCounter.most_common(commonStateLimit)])
    wordToState = {w: ("common" if w in mostCommonStates else "rare") for w in freqCounter}

    #newEmission = countEmission(text, wordToState)

    # if existt load file 
    oldStart = loadJSONFile("startCount.json") or {}
    oldTrans = loadJSONFile("transitionCount.json") or {}
    #oldEmission = loadJSONFile("emissionCount.json") or {}

    # then merge 
    mergedStart = mergeCount(Counter(oldStart), newStart)
    mergedTrans = mergeCounters(defaultdict(Counter, oldTrans), newTrans)
    #mergedEmission = mergeCounters(defaultdict(Counter, oldEmission), newEmission)

    # get prob 
    startProb = calcProbabilities(mergedStart)
    transProb = calcProbNested(mergedTrans)
    #emissionProb = calcProbNested(mergedEmission)

    # save to file 
    saveJSONFile(startProb, "startProb.json")
    saveJSONFile(transProb, "transitionProb.json")
    #saveJSONFile(emissionProb, "emissionProb.json")
    saveJSONFile(dict(mergedStart), "startCount.json")
    # chat helped wit this one liner 
    saveJSONFile({k: dict(v) for k, v in mergedTrans.items()}, "transitionCount.json")
    #saveJSONFile({k: dict(v) for k, v in mergedEmission.items()}, "emissionCount.json")

    return startProb, transProb#, emissionProb



    

#scripting use 
if __name__ == "__main__":
    sProb, tProb = main()
    print("start prob")
    #print(sProb)
    print("trans prob")
    #print(tProb)
    #print("emission prob")
    #print(eProb)
    


