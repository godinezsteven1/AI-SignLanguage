# This file is meant for the word segmentation of the NLP
# The goal of this file is to be able to split the words accurately and correctly. 
import math
import json
from collections import defaultdict

startProbFileName = "startProb.json"
transitionProbName = "transitionProb.json"

# Reading our start probabilities 
def loadStartProbabilities(fileName=startProbFileName): 
    with open(fileName, 'r') as file:
        return json.load(file)
    print(f"{fileName} read")

    
# Reading our transition probabilities and create nested dict for easy access
def loadTransitionProbabilitiesCreateNestedDict(fileName=transitionProbName): 
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
#runs it with (maybe corpus) dataset csv postgreSQl table with word: frequency 
def textSegmentation(text, wordProb=None):
    textLength = len(text)
    #tuple from -inf to text length for (word, probability) or technically (%, word)
    # if -inf = no valid seg found for now 
    table = (-math.inf, []) for _ in range(textLength + 1)
    #base case no string,g 
    table[0] = (0.0,[])

    for i in range(textLength):
        #current score and best score at i 
        currentScore = table[i]
        currentSegment = table[i]

        #has not been set up / does not exist. 
        if currentScore == -math.inf:
            continue

            
        #need to go to the word endings 
        for j in range(i + 1, n + 1):

        
        #try all combinations 
        #does word exist chcek from csv?
        #log word for probabilities 
        #make it score update to new score 
        #if new score better then curr score 
        #   update score and append to segmentation 
        #get best score/ segmentation 
        #return it 
    return True #optimalSegmentation
            
        
        
    
    
