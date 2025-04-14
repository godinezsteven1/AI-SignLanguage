# this file is meantt to fix spelling mistakes 
# uses dictionary or something to compare word  with spelling


from spellchecker import SpellChecker


inputLanguage = 'en'
#inputLanguage = 'es' #spanish
#inputLanguage = 'de' #german
def initSpellChecker(language=inputLanguage):
    spell = SpellChecker(language=language)
    return spell


# returns corrceted spelled word, or closest thing  
def correctWord(word, spell):
    if word in spell:
        return word
    return spell.correction(word)


def wordCorrection(segmentedWords,ISO):
    spell = initSpellChecker(ISO)
    # iterate throgh 
    return [correctWord(word, spell) for word in segmentedWords]


if __name__ == "__main__":
    testCases = [
        ["i", "go", "fsat"],               
        ["ths", "is", "a", "tst"],         
        ["lats", "go", "togeher"],         
        ["wht", "if", "we", "try", "ths"], 
        ["i", "lvoe", "it"],     
    ]
    
    for words in testCases:
        print(f"Original: {' '.join(words)}")
        corrected = wordCorrection(words, inputLanguage)
        print(f"Corrected: {' '.join(corrected)}")
