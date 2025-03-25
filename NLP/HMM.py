# This HMM class is made for probability and likilhood of word sequence
#given hidden states of user input 
# uses states observations emission probabilities for the HMM 
# uses Viterbi algorithm, a brute force algo, to find the most likely sequence of words 


class HMM:
    def __init__(self, states, startProb, transitionProb, emissionProb):
        self.states = states
        self.startProb = startProb
        self.transitionProb = transitionProb
        self.emissionProb = emissionProb

    #basic implementation of viterbi, tailor later
    def viterbi(self, obs):
        V = [{}] #probability at each step 
        path = {} #sequence 

        for y in self.states:
            V[0][y] = self.startProb[y] * self.emissionProb[y][obs[0]] #first observation 
            path[y] = [y]
        for t in range(1,len(obs)):
            V.append({})
            newPath = {}
            for y in self.states:
                # account for all previous 
                (prob, state) = max([(V[t - 1][y0] * self.transitionProb[y0][y] * self.emissionProb[y0][obs[t]], y0) for y0 in self.states])
                #store max prob 
                V[t][y] = prob
                newPath[y] = path[state] + [y] #updates path 
                path = newPath # updates path directory 
            (prob, state) = max([(V[-1][y], y) for y in self.states]) # state w highest prob in final observation 
            #return (prob, path[state]) # return prob w its state
            #print(prob, path[state]) # print 
        return path[state] # returns most likely obvservation sequence 
            
    #states = all states of model?
    #obs = user input
    #prob yet to be calculated 
                
#need to add start probabilities, transition probabilities, emissin probabilities with a reddit webscrapper
# needs to filter out NSFW textg 
