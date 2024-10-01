import torchtext
from collections import OrderedDict
import torchtext.vocab

START_TOKEN = '<start>'
END_TOKEN = '<end>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TOKENS = ['[#Branch1]', '[#Branch2]', '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[=As]', '[=Branch1]', '[=Branch2]', '[=C]', '[=N+1]', '[=N-1]', '[=NH1+1]', '[=NH2+1]', '[=N]', '[=O+1]', '[=O]', '[=P+1]', '[=PH1]', '[=P]', '[=Ring1]', '[=Ring2]', '[=S+1]', '[=SH1]', '[=S]', '[=Se+1]', '[=Se]', '[=Te+1]', '[Al]', '[As]', '[B-1]', '[BH1-1]', '[BH2-1]', '[BH3-1]', '[B]', '[Br]', '[Branch1]', '[Branch2]', '[C+1]', '[C-1]', '[CH1-1]', '[C]', '[Cl+3]', '[Cl]', '[F]', '[H]', '[I+1]', '[I]', '[N+1]', '[N-1]', '[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', '[N]', '[Na]', '[O+1]', '[O-1]', '[OH0]', '[OH1+1]', '[O]', '[P+1]', '[PH1]', '[P]', '[Ring1]', '[Ring2]', '[S+1]', '[S-1]', '[SH1]', '[S]', '[Se+1]', '[SeH1]', '[Se]', '[Si]', '[Te]']
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]


def get_vocab():
    vocab = torchtext.vocab.vocab(ordered_dict=OrderedDict({e:1 for e in TOKENS}), specials=SPECIAL_TOKENS)
    vocab.set_default_index(vocab[UNK_TOKEN])

    return vocab

