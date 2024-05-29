from src.Tools.LibRASR import Configuration
from src.Tools.LibRASR import Lexicon
from src.Tools.LibRASR import PhonemeInventory

help(Lexicon)
config = Configuration()
lex = Lexicon(config)
lemma = lex.new_lemma("banana")
print(lemma.name().str())
del lex
print(lemma.name().str())
