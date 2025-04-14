# This file is meant for the word segmentation of the NLP
# The goal of this file is to be able to split the words accurately and correctly. 
import math
import json
from collections import defaultdict

startProbFileName = "startProb.json"
transitionProbName = "transitionProb.json"

# get json 
def loadStartProbabilities(fileName=startProbFileName): 
    with open(fileName, 'r') as file:
        startProb = json.load(file)
        print(f"{fileName} read")
        return startProb

    
# get json and  create nested dict 
def loadTransitionProbabilities(fileName=transitionProbName): 
    with open(fileName, 'r') as file:
        transitions = json.load(file)
    print(f"{fileName} read")

    nestedDict = {}
    for word, nextWords in transitions.items():
        nestedDict[word] = defaultdict(lambda: 1e-10, nextWords) # chat helped w the nameless func 
    # so my computer and urs wont explode - aka if words dont exist because my dataset is from reddit + lightweight
    answer = defaultdict(lambda: defaultdict(lambda: 1e-10)) # give it small probability 
    answer.update(nestedDict)

    return answer









#uses Viterbi Algorithm to find most likely sequence 
def textSegmentation(text, wordProb=None, transProb=None):
    # load if not given aka prevent dumb mistake... again lol 
    if wordProb is None:
        wordProb = loadStartProbabilities()
    if transProb is None:
        transProb = loadTransitionProbabilities()

    text = text.lower() # jsut in case 
    textLength = len(text)

    #max segment size at a time 
    maxSegmentSize = 15 #characters 
    non_valid_score = float('-inf')
    # viterbi table format  (bestlogprob, bestsegmentation)
    table = [(non_valid_score, []) for _ in range(textLength + 1)] #  chat helped this line 
    table[0] = (0.0, [])  # our base is no string has 0 (log) prob 

    #forward
    # voter type of implementtion 
    for i in range(textLength):
        currentScore, currentSegmentation = table[i]
        # skip if nothing was found/ no exist 
        if currentScore == non_valid_score:
            continue
        # trying all words from 1 -brute force 
        for j in range(i + 1, min(i + maxSegmentSize + 1, textLength + 1)):
            word = text[i:j]

            wordExist = word in wordProb
            # need to incentivize word being split and not staying together 
            wordBonus = 0
            if wordExist:
                wordBonus = math.log(100)


            
            # getting the score for given word 
            wordProbScore = math.log(wordProb.get(word, 1e-10)) + wordBonus
            # if previous words add probability to it 
            transProbScore = 0
            if currentSegmentation:
                prevWord = currentSegmentation[-1]
                if prevWord in transProb and word in transProb[prevWord]:
                    transProbScore = math.log(transProb[prevWord][word])
                else:
                    transProbScore = math.log(1e-10)  # unknown transitions
                    
            newScore = currentScore + wordProbScore + transProbScore # if 0s nothing will change (for lin3 64, 72)
        # if better exist 
            if newScore > table[j][0]:
                table[j] = (newScore, currentSegmentation + [word])






        
    # if no exist just return given text -- else (87) we return best segmentaiton 
    if table[textLength][0] == non_valid_score:
        result = []
        i = 0 
        while i < textLength:
            bestWord = None
            bestScore = float('-inf')

            for j in range(i + 1, min(i + maxSegmentSize + 1, textLength + 1)):
                word = text[i:j]
                if word in wordProb and wordProb[word] > 0: #might be redundant
                    score = math.log(wordProb[word])
                    if score > bestScore:
                        bestScore = score
                        bestWord = word
            if bestWord:
                result.append(bestWord)
                i += len(bestWord)
            else: # none found  - slang issues
                # take one character and moving on 
                result.append(text[i])
                i += 1 
        return result

                               
    return table[textLength][1]


# test simple 
if __name__ == "__main__":
    testTexts = [
        "igofast",
        "iloveit",
        "whatif",
        "goaway",
        "buthow"
    ]
    
    for testText in testTexts:
        segmented = textSegmentation(testText)
        print(f"Original: {testText}")
        print(f"Segmented: {' '.join(segmented)}")
