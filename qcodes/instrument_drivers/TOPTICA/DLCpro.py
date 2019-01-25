import logging
import re
from functools import partial
from time import sleep, time

from qcodes import validators as vals
from qcodes import Instrument, InstrumentChannel, VisaInstrument
from qcodes import Parameter

log = logging.getLogger(__name__)

DLCpro_val_dict = {
    int: vals.Ints(),
    float: vals.Numbers(),
    str: vals.Strings(),
    bool: vals.Bool()
}

DLCpro_get_parser_dict = {
    int: int,
    float: float,
    str: lambda s: s.strip('\"'),
    bool: None
}

DLCpro_set_parser_dict = {
    int: None,
    float: None,
    str: lambda s: f'\"{s}\"',
    bool: None
}

def make_cmd(cmd_type, name, arg=None):
    # cmd_type in param-ref, param-set!, param-disp, exec
    # name is parameter or command name
    # arg 
    arg_str = f' {arg!s}' if arg else ''
    return f'({cmd_type!s} \'{name!s}{arg_str!s})'

def DLCpro_val(val_type):
    # val_type in int, float, bool, or tuple of the previous
    if type(val_type) is tuple:
        return Tuple((DLCpro_val_dict[v] for v in val_type))
    elif val_type in DLCpro_val_dict:
        return DLCpro_val_dict[val_type]
    else:
        raise ValueError(f"{val_type} not in {DLCpro_val_dict}")

def DLCpro_mapping(val_type):
    if val_type is bool:
        return {True: "#t", False: "#f"}
    else:
        return None

def make_get_parser(val_type):
    if type(val_type) is tuple:
        return lambda tup: (DLCpro_get_parser_dict[v](e)
                            for v, e in zip(val_type, tup))
    elif val_type in DLCpro_get_parser_dict:
        return DLCpro_get_parser_dict[val_type]
    else:
        raise ValueError(f"{val_type} not in {DLCpro_get_parser_dict}")
    
def make_set_parser(val_type):
    if type(val_type) is tuple:
        return lambda tup: (DLCpro_set_parser_dict[v](e)
                            for v, e in zip(val_type, tup))
    elif val_type in DLCpro_set_parser_dict:
        return DLCpro_set_parser_dict[val_type]
    else:
        raise ValueError(f"{val_type} not in {DLCpro_set_parser_dict}")


class Tuple(vals.Validator):
    """
    requires a tuple of values which validate their corresponding
    validator
    """

    def __init__(self, validators: tuple) -> None:
        self._validators = validators
        self._len = len(self._validators)

    def validate(self, value: tuple, context: str='') -> None:
        if len(value) != self._len:
            raise ValueError(
                f'{value!r} is invalid, length must be '
                f'{self._len}; {context}')

        for i in range(self._len):
                self._validators[i].validate(value[i], context)
        
    def __repr__(self) -> str:
        _ = ', '.join([f"{val!r}" for val in self._validators])
        return f'<Tuple({_})>'

	
class DLCproGenericError(Exception):
    pass


class DLCproReadOnlyParam(Parameter):
    def __init__(self, name, instrument, val_type, **kwargs):
        scheme_name = ':'.join(instrument.name.split('_')[1:])
        scheme_name += ':' + name.replace('_','-')

        super().__init__(
            name,
            get_cmd=lambda: instrument.ask(make_cmd('param-ref',
                                                    scheme_name)),
            get_parser=make_get_parser(val_type),
            val_mapping=DLCpro_mapping(val_type),
            vals=DLCpro_val(val_type),
            **kwargs)

        self._instrument = instrument


class DLCproReadWriteParam(Parameter):
    def __init__(self, name, instrument, val_type, **kwargs):
        scheme_name = ':'.join(instrument.name.split('_')[1:])
        scheme_name += ':' + name.replace('_','-')

        super().__init__(
            name,
            get_cmd=lambda: instrument.ask(make_cmd('param-ref',
                                                    scheme_name)),
            get_parser=make_get_parser(val_type),
            set_cmd=lambda val: instrument.ask(make_cmd('param-set!',
                                                        scheme_name,
                                                        val)),
            set_parser=make_set_parser(val_type),
            val_mapping=DLCpro_mapping(val_type),
            vals=DLCpro_val(val_type),
            **kwargs)

        self._instrument = instrument


class DLCproLaser1(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_submodule('ctl', DLCproLaser1Ctl(self, 'ctl'))
        self.add_submodule(
            'power_stabilization',
            DLCproLaser1PowerStab(self, 'power-stabilization'))


class DLCproLaser1Ctl(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='wavelength_act',
                           parameter_class=DLCproReadOnlyParam,
                           val_type=float,
                           label='wavelength',
                           unit='nm')

        self.add_parameter(name='wavelength_set',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='target wavelength',
                           unit='nm')

        self.add_parameter(name='state',
                           parameter_class=DLCproReadOnlyParam,
                           val_type=int)


class DLCproLaser1PowerStab(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='enabled',
                           parameter_class=DLCproReadWriteParam,
                           val_type=bool,
                           label='power stabilization state')

        self.add_submodule(
            'gain',
            DLCproLaser1PowerStabGain(self, 'gain'))

        self.add_parameter(name='sign',
                           parameter_class=DLCproReadWriteParam,
                           val_type=bool,
                           label='sign of the power stabilization action')
                           
        self.add_parameter(name='input_channel',
                           parameter_class=DLCproReadWriteParam,
                           val_type=int,
                           label='power stabilization input channel')
                           
        self.add_parameter(name='setpoint',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='power stabilization target power',
                           unit='mW')

        self.add_parameter(name='input_channel_value_act',
                           parameter_class=DLCproReadOnlyParam,
                           val_type=float,
                           label='power stabilization actual power',
                           unit='mW')

        self.add_parameter(name='state',
                           parameter_class=DLCproReadOnlyParam,
                           val_type=int,
                           label="power stabilization state")

        self.add_parameter(name='feedforward_enabled',
                           parameter_class=DLCproReadWriteParam,
                           val_type=bool,
                           label='power stabilization feedforward state')
                           
        self.add_parameter(name='feedforward_factor',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='power stabilization feedforward factor',
                           unit='V/mA')


class DLCproLaser1PowerStabGain(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='all',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='overall gain')

        self.add_parameter(name='p',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='proportional gain',
                           unit='mA/mW')

        self.add_parameter(name='i',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='integral gain',
                           unit='mA/mW/ms')

        self.add_parameter(name='d',
                           parameter_class=DLCproReadWriteParam,
                           val_type=float,
                           label='differential gain',
                           unit='mA.us/mW')


class DLCproStandby(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='enabled',
                           parameter_class=DLCproReadWriteParam,
                           val_type=bool,
                           label='standby target state')

        self.add_parameter(name='state',
                           parameter_class=DLCproReadOnlyParam,
                           val_type=int,
                           label="standby state")


class DLCpro(VisaInstrument):
    """
    QCodeS driver for TOPTICA DLCpro laser controller
    """

    def __init__(self, name, address, **kwargs):
        """
        Args:
            name: The name of the instrument
            address: The VISA resource name of the instrument
                (e.g. "tcpip0::192.168.1.2::7230::socket")
        """
        super().__init__(name, address, **kwargs)

        self.visa_handle.encoding = 'utf-8'
        self.visa_handle.read_termination = '\n> '
        self.visa_handle.write_termination = '\n'

        self.welcome_msg = self.read_raw()[2:]

        self.add_submodule('laser1', DLCproLaser1(self, 'laser1'))
        self.add_submodule('standby', DLCproStandby(self, 'standby'))

        self.connect_message()

    def get_idn(self):
        vendor = "TOPTICA"
        model = self.ask(make_cmd("param-ref",
                                  "system-type")).strip('\"')
        serial = self.ask(make_cmd("param-ref",
                                   "serial-number")).strip('\"')
        firmware = self.ask(make_cmd("param-ref",
                                     "fw-ver")).strip('\"')
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}

    def write_raw(self, cmd):
        """
        Override the low-level interface to ``visa_handle.write``.

        Args:
            cmd (str): The command to send to the instrument.
        """
        # the simulation backend does not return anything on
        # write
        log.debug("Writing to instrument %s: %r", self.name, cmd)
        if self.visabackend == 'sim':
            # if we use pyvisa-sim, we must read back the 'OK'
            # response of the setting
            resp = self.visa_handle.ask(cmd)
            if resp != 'OK':
                log.warning('Received non-OK response from instrument '
                            '%s: %r.', self.name, resp)
        else:
            self.ask_raw(cmd)

    def ask_raw(self, cmd):
        """
        Overriding the low-level interface to ``visa_handle.ask``.

        Args:
            cmd (str): The command to send to the instrument.

        Returns:
            str: The instrument's response.
        """
        log.debug("Querying instrument %s: %r", self.name, cmd)

        _, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)

        return self.read_raw()

    def read_raw(self):
        # regexp necessary to deal with unreliable message ends
        # end_msg_pattern = re.compile(r'(\r\n)?(OK|ERROR)$')

        end_str = self.visa_handle.read_termination
        end_msg_pattern = re.compile(end_str)
        end_msg_status = False
        _=[]
        while not end_msg_status:
            raw_chunk = self.visa_handle.read_raw()
            chunk = raw_chunk.decode(self.visa_handle.encoding, 'ignore')
            _.append(chunk)
            match = end_msg_pattern.search(chunk)
            if match is not None:
                end_msg_status = True
        msg = end_msg_pattern.sub('', end_str[-1].join(_))

        log.debug(f"Reading from instrument: {msg}")

        err_pattern = re.compile(r'Error:\s+(-\d+)\s+([\w\s]+)')
        match = err_pattern.search(msg)
        if match is not None:
            raise DLCproGenericError({"code":int(match.group(1)),
                                      "message":match.group(2)})

        return msg