from .nvidia_arch_lexicon import NvidiaArchLexicon
from .amd_arch_lexicon import AmdArchLexicon


def arch_lexicon_factory(arch_name):
    if arch_name == "nvidia":
        return NvidiaArchLexicon()
    elif arch_name == "amd":
        return  AmdArchLexicon()
