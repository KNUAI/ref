import random
from nltk.corpus import wordnet

# Synonym Replacement (유의어 교체)
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    
    for random_word in random_word_list[:n]:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
    
    return ' '.join(new_words)

sentence = "The quick brown fox jumps over the lazy dog."
print(synonym_replacement(sentence, n=2))

# Random Insertion (랜덤 삽입)
def random_insertion(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        synonyms = []
        while not synonyms:
            random_word = random.choice(words)
            synonyms = wordnet.synsets(random_word)
        random_synonym = random.choice(synonyms).lemmas()[0].name()
        random_index = random.randint(0, len(words))
        words.insert(random_index, random_synonym)
    return ' '.join(words)

print(random_insertion(sentence, n=2))

# Random Deletion (랜덤 삭제)
def random_deletion(sentence, p=0.2):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(new_words if new_words else words)

print(random_deletion(sentence, p=0.3))

# Random Swap (랜덤 교체)
def random_swap(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

print(random_swap(sentence, n=2))
