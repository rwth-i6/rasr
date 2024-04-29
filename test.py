from src.Tools.LibRASR import Configuration
from src.Tools.LibRASR import Lexicon
from src.Tools.LibRASR import PhonemeInventory

help(Lexicon)
config = Configuration()
lex = Lexicon(config)
lex.set_phoneme_inventory(PhonemeInventory())
phon = lex.phoneme_inventory()
print(phon.num_phonemes())
lex.log_statistics()
