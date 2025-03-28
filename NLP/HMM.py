# This HMM class is made for probability and likilhood of word sequence
#given hidden states of user input 
# uses states observations emission probabilities for the HMM 
# uses Viterbi algorithm, a brute force algo, to find the most likely sequence of words 

import json


with open("startProb.json", "r") as f:
    startProb = json.load(f)

with open("transitionProb.json", "r") as f:
    transitionProb = json.load(f)

with open("emissionProb.json", "r") as f:
    emissionProb = json.load(f)





class HMM:
    def __init__(self, states, startProb, transitionProb, emissionProb):
        self.states = states
        self.startProb = startProb
        self.transitionProb = transitionProb
        self.emissionProb = emissionProb



    def viterbi(self, obs):
        T = len(obs) #all of our obvservations 
        V = [{}]  # V[t] holds the maximum probability for each state at time t.
        path = {} # highest probabilty seq

        for i in self.states:
            # get emission, if no exist get small number because prob not exist 
            emitProb = self.emissionProb[i].get(obs[0], 1e-10)
            # multiply the starting probability of state by emission if no exist small num 
            V[0][i] = self.startProb.get(i, 1e-10) * emitProb # pi(S_i) * b_i(O_1) (viterbi equation)
            # = small * small = unlikely sequence
            # = big * big = likely sequence 
            # local best 
            path[i] = [i]

        for t in range(1, T): #for all of our iterations
            V.append({}) #new dict 
            newPath = {}
            for i in self.states:
                # max path that ends in i (because its word word word i) with respect to
                # max and get best path
                (prob, prevState) = max((V[t-1][i0] * self.transitionProb.get(i0, {}).get(i, 1e-10) * #S_i -> S_j
                     self.emissionProb[i0].get(obs[t], 1e-10), i0)  #emitting O_j at S_j
                    for i0 in self.states) # chat helped write logic from the max to down here 
                #store givne max
                V[t][i] = prob
                # add current state i 
                newPath[i] = path[prevState] + [i]
            #update 
            path = newPath

        # geter for highest probability
        (finalProb, finalState) = max((V[T-1][i], i) for i in self.states)
        print(finalProb,path[finalState])
        return (finalProb, path[finalState])



if __name__ == "__main__":
    states = list(emissionProb.keys())
    HMModel = HMM(states, startProb, transitionProb, emissionProb)
    #constructing sentence with high probabilities given start, trans and emission 
    obs = ["people", "on", "reddit"]

    (prob, bestPath) = HMModel.viterbi(obs)
    print("obs sequence", obs)
    print("best path", bestPath)
    print("best path prob", prob)

#returns
#2.6160889470241992e-23 ['people', 'reddit', 'is']
#obs sequence ['people', 'on', 'reddit']
#best path ['people', 'reddit', 'is']
#best path prob 2.6160889470241992e-23

#1. needs to be higher probability --> dataset is not big enough probabilities are small 
#2. needs to return people on reddit, sequence was not a incorrect one --> dataset is not big enough 
