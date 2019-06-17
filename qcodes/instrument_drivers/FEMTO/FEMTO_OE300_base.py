import logging

import numpy as np

from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Enum

from typing import NamedTuple

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

        
class OE300State(NamedTuple):
    gain_mode: str
    gain: int
    lp_filter_bw: str
    coupling: str


class OE300Base(Instrument):
    """
    A driver for the FEMTO OE300 photodiode, controlled manually.
    """

    def __init__(self, name, cal_path=None, prefactor=1, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('prefactor',
                           label='Prefactor',
                           parameter_class=ManualParameter,
                           units=None,
                           initial_value=prefactor)

        if cal_path:
            self.load_cal_file(cal_path)
        else:
            self.cal = None

    def load_cal_file(self, path):
        raw_cal = np.genfromtxt(path,
                          delimiter=',', 
                          skip_header=1).T
        self.cal = {'wl_nm': raw_cal[0], 'A_W': raw_cal[1]}

    def set_state(self, state: OE300State):
        self.gain_mode.set(state.gain_mode)
        self.gain.set(state.gain)
        self.lp_filter_bw.set(state.lp_filter_bw)
        self.coupling.set(state.coupling)

    def get_state(self, state: OE300State):
        self.gain_mode.set(state.gain_mode)
        self.gain.set(state.gain)
        self.lp_filter_bw.set(state.lp_filter_bw)
        self.coupling.set(state.coupling)

        return OE300State(self.gain_mode.get(), self.gain.get(), self.lp_filter_bw.get(),self.coupling.get())
        