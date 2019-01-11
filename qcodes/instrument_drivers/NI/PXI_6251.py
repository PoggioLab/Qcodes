try:
    import nidaqmx
except ImportError:
    raise ImportError('Could not find nidaqmx module.')

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel


class PXI_6251(Instrument):
    pass