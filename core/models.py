from .BANet import BANet
from .MAFNet import MAFNet


def build_model(name, args):
    name = name.lower()
    if name == 'banet':
        return BANet(args)
    if name == 'mafnet':
        return MAFNet(args, use_affa=True, use_aahf=True)
    if name == 'affa_conv':
        return MAFNet(args, use_affa=True, use_aahf=False)
    if name == 'aahf_only':
        return MAFNet(args, use_affa=False, use_aahf=True)
    raise ValueError(f'Unknown model {name!r}')
