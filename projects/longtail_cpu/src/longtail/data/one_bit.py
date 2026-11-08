"""Single-component presence pairs; shared noise is not the sample identity."""
import numpy as np
from .signals import Simulator
from .v3_fresh import compose_v3

_SIM = None


def initialize(physics):
    global _SIM
    _SIM = Simulator(physics)


def physical_descriptor(iq):
    """Single-view amplitude-normalized spectral/temporal descriptors, no truth."""
    x = np.asarray(iq); x = x - x.mean()
    x = x / np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x)))**2
    bands = np.array([v.mean() for v in np.array_split(spectrum, 32)])
    bands = np.log1p(bands / max(spectrum.mean(), 1e-12))
    acf = [abs(np.vdot(x[:-lag], x[lag:])) / (len(x)-lag) for lag in [1,2,4,8,16,32,64,128]]
    envelope = np.abs(x)
    stats = np.r_[np.quantile(envelope, [.1,.25,.5,.75,.9,.99]), np.mean(envelope**4)]
    return np.r_[bands, acf, stats].astype(np.float32)


def pair(simulator, background_code, target, seed, classes):
    if not 0 < background_code < 64 or (background_code >> target) & 1:
        raise ValueError('One-bit background must be nonempty and omit the target')
    active = np.flatnonzero(((background_code | (1 << target)) >> np.arange(6)) & 1)
    plus = compose_v3(simulator, [classes[c] for c in active], seed, return_components=True)
    position = int(np.flatnonzero(active == target)[0]); component = plus['clean_components'][position]
    negative = plus['iq'] - component
    error = float(np.linalg.norm((plus['iq']-negative)-component) / np.linalg.norm(component))
    assert error < 1e-12
    return active, plus, negative, position, error


def chunk_worker(task):
    backgrounds, targets, seeds, edges, classes = task
    n = len(seeds)*2; block = {'iq_complex': np.empty((_SIM.length,n),np.complex128),
        'label_selected':np.zeros((6,n),np.uint8), 'receiver_noise_seed':np.empty((1,n),np.uint32),
        'sample_seed':np.repeat(np.asarray(seeds,np.uint32),2)[None,:],
        'sample_id':((1<<40)+2*np.repeat(np.asarray(seeds,np.int64),2)+np.tile([0,1],len(seeds)))[None,:],
        'pair_id':np.repeat(np.asarray(seeds,np.int64),2)[None,:],
        'pair_target':np.repeat(np.asarray(targets,np.int8),2)[None,:],
        'edge_id':np.repeat(np.asarray(edges,np.int16),2)[None,:],
        'pair_side':np.tile([0,1],len(seeds)).astype(np.uint8)[None,:],
        'raw_difference_relative_error':np.empty((1,n)),
        'weak_component_mask':np.zeros((6,n),np.uint8), 'physical_descriptor':np.empty((47,n),np.float32)}
    for name in ['component_effective_JNR_dB','component_measured_power_dB','relative_power_dB','component_seeds']:
        block[name]=np.full((6,n),np.nan)
    for i,(background,target,seed) in enumerate(zip(backgrounds,targets,seeds)):
        active,plus,minus,position,error=pair(_SIM,int(background),int(target),int(seed),classes)
        for side,iq in enumerate([minus,plus['iq']]):
            j=2*i+side; keep=np.arange(len(active))!=position if side==0 else np.ones(len(active),bool)
            selected=active[keep];power=plus['power'][keep];relative=power-power.max()
            block['iq_complex'][:,j]=iq;block['label_selected'][selected,j]=1
            block['receiver_noise_seed'][0,j]=plus['noise_seed'];block['raw_difference_relative_error'][0,j]=error
            block['physical_descriptor'][:,j]=physical_descriptor(iq)
            for name,values in [('component_effective_JNR_dB',plus['jnr'][keep]),('component_measured_power_dB',power),
                                ('relative_power_dB',relative),('component_seeds',np.asarray(plus['component_seeds'])[keep])]:
                block[name][selected,j]=values
            block['weak_component_mask'][selected,j]=(len(selected)>1)&(relative<=-10+1e-6)
    return block
