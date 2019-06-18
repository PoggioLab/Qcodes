import logging

from ctypes import CDLL, byref, c_int, create_string_buffer, sizeof

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
        old_raw_value = self._raw_value

        self._raw_value = self.value_to_raw_value(value)
        try:
            self._instrument.write_data()
        except OE300Error as e:
            self._raw_value = old_raw_value
            raise e
            
    def value_to_raw_value(self, value):
        return self.vals._valid_values.index(value)
        
    def raw_value_to_value(self, raw_value):
        return self.vals._valid_values[raw_value]

    def make_bits(self):
        return f'{self._raw_value:0{self._nbits}b}'


class OE300GainMode(OE300BaseParam):    
    def set_raw(self, value): # pylint: disable=method-hidden     
        old_gain_vals = self._instrument.gain.vals
        gains = LOW_NOISE_GAINS if value == 'L' else HIGH_SPEED_GAINS
        self._instrument.gain.vals = Enum(*gains)
        try:
            super().set_raw(value)
        except OE300Error as e:
            self._instrument.gain.vals = old_gain_vals
            raise e


class OE300LUCI(OE300Base):
    """
    A driver for the FEMTO OE300 photodiode, controlled through the LUCI-10 interface. The LUCI-10 dll needs to be installed.
    """
    dll_path = 'C:\\Program Files (x86)\\FEMTO\\LUCI-10\\Driver\\LUCI_10_x64.dll'

    def __init__(self, name, index=None, idn=None, dll_path=None, cal_path=None, prefactor=1, **kwargs):
        super().__init__(name, cal_path, prefactor, **kwargs)

        log.info('Loading LUCI-10 dll')
        self.LUCI = CDLL(dll_path or self.dll_path)

        log.info('Connecting to OE300 device')
        # connect to desired device
        idn_tmp = c_int()
        dev_idn_list = []
        for index in range(1, self.LUCI.EnumerateUsbDevices() + 1): #index starts at 1.
            self.LUCI.ReadAdapterID(index, byref(idn_tmp))
            dev_idn_list.append(idn_tmp)

        if index is None and idn is None:
            self._index=1
        elif index is None and idn is not None:
            self._index = dev_idn_list.index(idn) + 1 # index starts at 1.
        elif index is not None and idn is None:
            self._index = index
        else:
            if index == dev_idn_list.index(idn) + 1: # index starts at 1.
                self._index = index
            else:
                raise ValueError("index and idn do not match, it is best to only specify one")

        log.info('Reseting OE300 device')
        # reset device
        self.LUCI.WriteData(self._index, 0, 0)

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

        log.info('LUCI-controlled OE300 initialization complete')

    def write_data(self):        
        low_byte = int(self.lp_filter_bw.make_bits() +
                       self.gain_mode.make_bits() +
                       self.coupling.make_bits() +
                       self.gain.make_bits(), 2)
        error_code = self.LUCI.WriteData(self._index, low_byte, 0)
        if error_code:
            raise OE300Error(error_code)

    def get_idn(self):
        p = create_string_buffer(50)
        self.LUCI.GetProductString(self._index, p, sizeof(p))

        vendor = 'FEMTO'
        model = p.value.decode()
        serial = None
        firmware = None
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}
