from src.Tools.LibRASR import Configuration
from src.Tools.LibRASR import Lexicon

config = Configuration()
lex = Lexicon(config)
lex.load("output.xml")
lex.log_statistics()
lex.write_xml()
