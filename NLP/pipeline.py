
import os
from wordSegm import textSegmentation
from wordCorrection import wordCorrection
from HMM import HMM

def NLPpipeline(text, language=None):
    
    # fail safe 
    if language is None:
        language = "en"  # default to English 
        
    # 1)  Word Segmentation
    segmentedWords = textSegmentation(text, language)
    #print(f" segmentation: {' '.join(segmentedWords)}")
    
    # 2) Word Correction
    try:
        correctedWords = wordCorrection(segmentedWords, language)
        if correctedWords is None: # if terrible spelling or nothing exist (slang) return itself 
            correctedWords = segmentedWords
        
        #avoid join errors, none afterwards --> empty ""
        correctedWords = [word if word is not None else "" for word in correctedWords] # chhat wrote this line 
        # print(f" correction: {' '.join(correctedWords)}")
    except Exception as e:
        print(f"Correction error: {e}")
        correctedWords = segmentedWords
    
    # 3) HMM - Bigram 
    hmm = HMM()
    hmm.loadModel(language)
    
    probability, bestPath = hmm.viterbi(correctedWords)
    # print(f"HMM path: {' '.join(bestPath)}")
    #print(f"path probability: {probability}")
    
    return correctedWords, bestPath, probability


if __name__ == "__main__":
    rawWords = [
        "latsgo",
        "igofst",
        "iloveyo"
    ]
    
    for test in rawWords:
        print(f"\n '{test}'")
        corrected, path, prob = NLPpipeline(test)
        print(f" result: '{' '.join(corrected)}'")
