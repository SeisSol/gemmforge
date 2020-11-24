from .nvidia_arch_dictionary import NvidiaArchDictionary
from .amd_arch_dictionary import AmdArchDictionary


def arch_dictionary_factory(arch_name):
    if arch_name == "nvidia":
        return NvidiaArchDictionary()
    elif arch_name == "amd":
        return  AmdArchDictionary()
