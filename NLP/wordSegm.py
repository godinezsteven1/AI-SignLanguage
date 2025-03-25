# This file is meant for the word segmentation of the NLP
# The goal of this file is to be able to split the words accurately and correctly. 
import math

#uses Viterbi Algorithm to find most likely sequence 
#runs it with (maybe corpus) dataset csv postgreSQl table with word: frequency 
def textSegmentation(text, wordProb):
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
            
        
        
    
    
