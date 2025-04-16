# This file is meant for the word segmentation of the NLP
# The goal of this file is to be able to split the words accurately and correctly. 
import math
import json
from collections import defaultdict
import os 

startProbFileName = "startProb.json"
transitionProbName = "transitionProb.json"


supportedLanguages = ["en", "es", "de"]
def getLangPath(fileName, language=None):
    if language is None or language not in supportedLanguages:
    #base case fall back  -- english 
        language = "en" 
    directory = os.path.join("NLP", "Languages", language)
    #another fall back make dir if no exist, # debug too 
    os.makedirs(directory,exist_ok=True)
    return os.path.join(directory, fileName)




# get json 
def loadStartProbabilities(language=None): 
    fileName = getLangPath(startProbFileName)
    try:
        with open(fileName, 'r') as file:
            startProb = json.load(file)
            # debug 
            #print(f"{fileName} is being read")
            return startProb
    except Exception as e:
            #print(f"Error with {fileName} probably does not exist")
        return {} # to prevent ocmputer for exploding
    

    
# get json and  create nested dict 
def loadTransitionProbabilities(language=None): 
    smallProbs = lambda: defaultdict(lambda: 1e-10)
    # built in failsafe --> En
    fileName = getLangPath(transitionProbName, language)
    try:
        with open(fileName, 'r') as file:
            transitions = json.load(file)
        print(f"{fileName} is being read")
        nestedDict = {} 
        for word, nextWords in transitions.items():
            nestedDict[word] = defaultdict(lambda: 1e-10, nextWords) # chat helped
        answer = defaultdict(smallProbs) # give it small probability 
        answer.update(nestedDict)
        return answer
    except Exception as e:
        print(f"Error loading {fileName}: {e}")
        return defaultdict(smallProbs)
        



#uses Viterbi Algorithm to find most likely sequence 
def textSegmentation(text, language=None, wordProb=None, transProb=None):
    # load if not given aka prevent dumb mistake... again lol 
    if wordProb is None:
        wordProb = loadStartProbabilities(language)
    if transProb is None:
        transProb = loadTransitionProbabilities(language)

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
        result = [] # pssoibe issue 
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

    print("Default English Test")
    for testText in testTexts:
        segmented = textSegmentation(testText)
        print(f"Original: {testText}")
        print(f"Segmented: {' '.join(segmented)}")

    print("Targeted English Test")
    for testText in testTexts:
            segmented = textSegmentation(testText, language="en")
            print(f"Original: {testText}")
            print(f"Segmented: {' '.join(segmented)}")
