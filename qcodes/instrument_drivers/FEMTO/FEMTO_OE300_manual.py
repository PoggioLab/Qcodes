import logging

from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils.validators import Enum

from .FEMTO_OE300_base import (OE300State, OE300Error, LOW_NOISE_GAINS, HIGH_SPEED_GAINS,
                               LP_SETTINGS, COUPLING_MODES, GAIN_SETTINGS, ERROR_TABLE,
                               OE300Base)

log = logging.getLogger(__name__)


class OE300BaseParam(Parameter):
    def __init__(self, name, instrument, vals, nbits, **kwargs):
        super().__init__(name=name, instrument=instrument, vals=vals, **kwargs)
        self._raw_value = 0
        self._nbits = nbits

    def get_raw(self): # pylint: disable=method-hidden
        return self.raw_value_to_value(self._raw_value)

    def set_raw(self, value): # pylint: disable=method-hidden
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


class OE300Manual(OE300Base):
    """
    A driver for the FEMTO OE300 photodiode, controlled manually.
    """

    def __init__(self, name, cal_path=None, prefactor=1, **kwargs):
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

        self.add_parameter('prefactor',
                           label='Prefactor',
                           parameter_class=ManualParameter,
                           units=None,
                           initial_value=prefactor)

        log.info('Manually controlled  OE300 initialization complete')

    def get_idn(self):

        vendor = 'FEMTO'
        model = None
        serial = None
        firmware = None
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}
