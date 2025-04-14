# This HMM class is made for probability and likilhood of word sequence
# given hidden states of user input 
# uses states observations, and transitions for the Bigram HMM 
# uses Viterbi algorithm, a brute force algo, to find the most likely sequence of words 

import json
import math
from collections import defaultdict
import os

startFileName="startProb.json"
transFileName="transitionProb.json"

def getLangPath(filename, language=None):
    if language is None or language not in ["en", "es", "de"]: # english, spanish, german 
        language = "en"  #fall back to english 
    language_dir = os.path.join("Languages", language)
    os.makedirs(language_dir, exist_ok=True) #creates dir if no exist, kind of nice debugger
    return os.path.join(language_dir, filename)

class HMM:
    def __init__(self):
        self.startProb = defaultdict(float)
        #dictionary of dictionary
        self.transitionProb = defaultdict(lambda: defaultdict(float))
    
    def train(self, sequence):
        startCount = defaultdict(int) #track start and transition probabilities 
        transitionCount = defaultdict(lambda: defaultdict(int))

        for freq in sequence:
            if not freq:
                continue
            startCount[freq[0]] += 1 #initialize first word as start word
            for i in range(1, len(freq)):
                prev = freq[i - 1]
                curr = freq[i]  # fixed from seq[i]
                transitionCount[prev][curr] += 1 #count transitions between consecutive words

        totalStart = sum(startCount.values()) # add al probs 
        for word, count in startCount.items(): # for all counts and words
            self.startProb[word] = count / totalStart # get prob for start words

        for prev in transitionCount:
            totalTrans = sum(transitionCount[prev].values()) # now with transitions
            for w in transitionCount[prev]:
                # get transitions probabilities 
                self.transitionProb[prev][w] = transitionCount[prev][w] / totalTrans

    #opens file and READS - updated to use language parameter
    def loadModel(self, language=None, startFile=None, transFile=None):
        # If specific files are not provided, use language path
        if startFile is None:
            startFile = getLangPath(startFileName, language)
        if transFile is None:
            transFile = getLangPath(transFileName, language)
            
        try:
            with open(startFile, 'r') as file: #read
                self.startProb = defaultdict(float, json.load(file))
                print(f"Reading file {startFile}")
            with open(transFile, 'r') as file:
                rawTransData = json.load(file)
                self.transitionProb = defaultdict(lambda: defaultdict(float))
                print(f"Reading file {transFile}")
                for prev, curr in rawTransData.items():
                    self.transitionProb[prev] = defaultdict(float, curr)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # Viterbi remains unchanged - it uses the loaded probabilities
    def viterbi(self, obs):
        if not obs:
            return float('-inf'), []
        
        V = [{}]  # V[t] holds the maximum probability for each state at time t.
        path = {} # highest probabilty seq

        firstObs = obs[0]

        for word in self.transitionProb:
            # log for prob since can be small
            # get word else return small prob if no exist .get(word, #)
            V[0][word] = math.log(self.startProb.get(word, 1e-10))
            path[word] = [word]

        for t in range(1, len(obs)):
            V.append({})
            newPath = {}
            currObs = obs[t] #obs at time t
            
            for prevW in V[t-1]: # from previous time step 
                # get probability of our current path, log it because small prob
                # previous word + log of transition prob of prev word going to curr obvservation, if no exist get
                # a very small probability 
                prob = V[t - 1][prevW] + math.log(self.transitionProb[prevW].get(currObs, 1e-10))
                
                if currObs not in V[t] or prob > V[t][currObs]:
                    V[t][currObs] = prob
                    newPath[currObs] = path[prevW] + [currObs]
                
            path = newPath

        # find the path with the highest final log probability in the end 
        n = len(obs) - 1
        if V[n]:
            bestPath = max(V[n], key=V[n].get)
                #V[n] returns high prob path, path[bestPath] returns said path
                #kind of genius if you think about this if statement 
            return V[n][bestPath], path[bestPath]
        else:
            return float('-inf'), [] # so my code does not explode 

if __name__ == "__main__":
    # Test with default language (English)
    hmm_en = HMM()
    hmm_en.loadModel()  # Defaults to English

    obs = ["what", "is", "my", "name"]
    prob, bestPath = hmm_en.viterbi(obs)

    print("English test:")
    print("Observation sequence:", obs)
    print("Best path:", bestPath)
    print("Best path probability:", prob)
    
   
