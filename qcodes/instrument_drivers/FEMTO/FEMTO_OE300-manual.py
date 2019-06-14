import logging

from ctypes import CDLL, byref, c_int, create_string_buffer, sizeof

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Enum

log = logging.getLogger(__name__)

LOW_NOISE_GAINS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
HIGH_SPEED_GAINS = (1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
LP_SETTINGS = ('FBW', '10MHz', '1MHz') # ordered by corresponding binary coding 
COUPLING_MODES = ('DC', 'AC')
GAIN_SETTINGS = ('L', 'H')

ERROR_TABLE = {-1: "Invalid index: selected LUCI-10 not in list",
               -2: "Instrument error: LUCI-10 does not respond"}

class OE300Error(Exception):
    def __init__(self, error_code):
        super().__init__(ERROR_TABLE[error_code])


class OE300BaseParam(Parameter):
    def __init__(self, name, instrument, vals, nbits, **kwargs):
        super().__init__(name=name, instrument=instrument, vals=vals, **kwargs)
        self._raw_value = 0
        self._nbits = nbits

    def get_raw(self): # pylint: disable=method-hidden
        return self.raw_value_to_value(self._raw_value)

    def set_raw(self, value): # pylint: disable=method-hidden
        old_raw_value = self._raw_value

        self._raw_value = self.value_to_raw_value(value)
            
    def value_to_raw_value(self, value):
        return self.vals._valid_values.index(value)
        
    def raw_value_to_value(self, raw_value):
        return self.vals._valid_values[raw_value]

    def make_bits(self):
        return f'{self._raw_value:0{self._nbits}b}'


class OE300GainMode(OE300BaseParam):    
    def set_raw(self, value): # pylint: disable=method-hidden 
        gains = LOW_NOISE_GAINS if value == 'L' else HIGH_SPEED_GAINS
        self._instrument.gain.vals = Enum(*gains)
        super().set_raw(value)


class OE300Manual(Instrument):
    """
    A driver for the FEMTO OE300 photodiode, controlled manually.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('gain',
                           label='Gain',
                           vals=Enum(*LOW_NOISE_GAINS),
                           nbits=3,
                           parameter_class=OE300BaseParam)

        self.add_parameter('coupling',
                           label='Coupling',
                           vals=Enum(*COUPLING_MODES),
                           nbits=1,
                           parameter_class=OE300BaseParam)

        self.add_parameter('gain_mode',
                           label='Gain mode',
                           vals=Enum(*GAIN_SETTINGS),
                           nbits=1,
                           parameter_class=OE300GainMode)

        self.add_parameter('lp_filter_bw',
                           label='Lowpass filter bandwidth',
                           vals=Enum(*LP_SETTINGS),
                           nbits=2,
                           parameter_class=OE300BaseParam)

        log.info('Manually controlled  OE300 initialization complete')

    def get_idn(self):

        vendor = 'FEMTO'
        model = None
        serial = None
        firmware = None
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}
