from functools import partial
from qcodes import Instrument, InstrumentChannel, VisaInstrument
from qcodes import ArrayParameter
from qcodes.tests.instrument_mocks import setpoint_generator
from qcodes.utils import validators as vals

import numpy as np
import struct
import re

multiplier_dict = {
    'P': 1e-12,
    'N': 1e-9,
    'U': 1e-6,
    'M': 1e-3,
    'K': 1e3
}

def ans_parser(ans, unit=None, parser=str):
            """
            Parse "{value} [{multiplier}{unit}]" type answers from TD44MXsB.

            Args:
                ans: The answer from the instrument
                unit: The expected unit(s). String of list of strings. Defaults to None.
                parser: Function to use to parse the value.

            Returns parser(value).
            """
            ans = ans.strip()
            if not ans:
                raise ValueError('Empty answer.')

            if not unit: #None or empty string                
                val = ans
            else:
                ans_pattern = re.compile(rf'(.+) ([A-Z]*){unit}')
                ans_match = ans_pattern.search(ans)
                if ans_match is not None:
                    val, modifier = ans_match.group(1,2)
                    val = float(val)
                    val *= 1 if not modifier else multiplier_dict[modifier]

                else:
                    raise ValueError(f'Unexpected answer {ans!r}.')

            if parser is int:
                val = float(val)

            return parser(val)


class TraceNotReady(Exception):
    pass


class TraceSetPointsChanged(Exception):
    pass


class WaveformArraySeq(ArrayParameter):
    """
    Will return a waveform from oscilloscope
    """

    def __init__(self, name, instrument, channel_id):
        super().__init__(name,
                         shape=(1,1024),
                         label='Voltage',
                         unit='V',
                         setpoint_names=('Seq','Time'),
                         setpoint_labels=(
                             'sequence number',
                             '{} time series'.format(channel_id)),
                         setpoint_units=('','s'),
                         docstring='raw waveform from the scope',
                         )

        self._channel_id = channel_id
        self._instrument = instrument

    def prepare_waveform(self, seq_setpoints=None, seq_name=None, seq_label=None,
                         seq_unit=None):
        """
        Prepare the scope for returning waveform data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed

        # shorthand
        instr = self._instrument

        # acquire waveform description
        instr.write("{}:WF? DESC".format(self._channel_id))
        wavedesc = instr._parent._read_ieee_block()

        # extract Time setpoints info
        self.Nsamples, = struct.unpack_from('l', wavedesc, 116)
        self.Nseqs, = struct.unpack_from('l', wavedesc, 144)
        self.horz_dt, self.horz_offset = struct.unpack_from('fd', wavedesc, 176)
        self.Npts = int(self.Nsamples/self.Nseqs)
        self.time_setpoints = tuple(self.horz_offset + self.horz_dt
                                    * np.arange(self.Npts))

        # set setpoints info
        if ((seq_setpoints is not None) and (len(seq_setpoints) != self.Nseqs)):
            raise ValueError('Length of seq_setpoints {} does \
                              not match number of sequences {}'.format(
                                  len(seq_setpoints), self.Nseqs))
        self.seq_setpoints = (tuple(seq_setpoints)
                              if seq_setpoints
                              else tuple(range(self.Nseqs)))

        if seq_name is not None:
            self.setpoint_names = (seq_name,'Time')

        if seq_unit is not None:
            self.setpoint_units = (seq_unit,'s')

        if seq_label is not None:
            self.setpoint_labels = (seq_label,
                                   f'{self._channel_id} time series')

        self.setpoints = setpoint_generator(self.seq_setpoints, self.time_setpoints)
        self.shape = (self.Nseqs, self.Npts)

        # make this on a per channel basis?
        self._instrument._parent.trace_ready = True

    def get_raw(self):
        # when get is called the setpoints have to be known already
        # (saving data issue). Therefor create additional prepare function that
        # queries for the size.
        # check if already prepared
        if not self._instrument._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # shorthand
        instr = self._instrument

        instr.write("{}:WF?".format(self._channel_id))
        wave = instr._parent._read_ieee_block()

        comm_type, comm_order = struct.unpack_from('??', wave, 32)
        dat1_offset = np.sum(struct.unpack_from('6l', wave, 36))
        Nsamples, = struct.unpack_from('l', wave, 116)
        Nseqs, = struct.unpack_from('l', wave, 144)
        vert_gain, vert_offset = struct.unpack_from('ff', wave, 156)
        horz_dt, horz_offset = struct.unpack_from('fd', wave, 176)

        if ((Nsamples,Nseqs,horz_dt,horz_offset) 
            != (self.Nsamples,self.Nseqs,self.horz_dt,self.horz_offset)):
            raise TraceSetPointsChanged('prepared setpoints do not match \
                                        with waveform, run prepare_waveform')

        Npts = int(Nsamples/Nseqs)

        dat1_array = np.ndarray((Nseqs,Npts),
                                'h' if comm_type else 'b',
                                wave, dat1_offset)
        # scale y values
        out_array = vert_offset + vert_gain * dat1_array

        return out_array


class WaveformArray(ArrayParameter):
    """
    Will return a waveform from oscilloscope
    """

    def __init__(self, name, instrument, channel_id):
        super().__init__(name,
                         shape=(1024,),
                         label='Voltage',
                         unit='V',
                         setpoint_names=('Time',),
                         setpoint_labels=(f'{channel_id} time series',),
                         setpoint_units=('s',),
                         docstring='raw waveform from the scope',
                         )

        self._channel_id = channel_id
        self._instrument = instrument

    def prepare_waveform(self):
        """
        Prepare the scope for returning waveform data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed

        # shorthand
        instr = self._instrument

        # acquire waveform description
        instr.write("{}:WF? DESC".format(self._channel_id))
        wavedesc = instr._parent._read_ieee_block()

        # extract Time setpoints info
        self.Nsamples, = struct.unpack_from('l', wavedesc, 116)
        self.Nseqs, = struct.unpack_from('l', wavedesc, 144)
        self.horz_dt, self.horz_offset = struct.unpack_from('fd', wavedesc, 176)
        self.Npts = int(self.Nsamples/self.Nseqs)
        self.time_setpoints = tuple(self.horz_offset + self.horz_dt
                                    * np.arange(self.Npts))

        # set setpoints info
        if (self.Nseqs != 1):
            raise ValueError(f'Number of sequences {self.Nseqs} should'
                             'be 1.')

        self.setpoints = (self.time_setpoints,)
        self.shape = (self.Npts,)

        # make this on a per channel basis?
        self._instrument._parent.trace_ready = True

    def get_raw(self):
        # when get is called the setpoints have to be known already
        # (saving data issue). Therefor create additional prepare function that
        # queries for the size.
        # check if already prepared
        if not self._instrument._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # shorthand
        instr = self._instrument

        instr.write("{}:WF?".format(self._channel_id))
        wave = instr._parent._read_ieee_block()

        comm_type, comm_order = struct.unpack_from('??', wave, 32)
        dat1_offset = np.sum(struct.unpack_from('6l', wave, 36))
        Nsamples, = struct.unpack_from('l', wave, 116)
        Nseqs, = struct.unpack_from('l', wave, 144)
        vert_gain, vert_offset = struct.unpack_from('ff', wave, 156)
        horz_dt, horz_offset = struct.unpack_from('fd', wave, 176)

        if ((Nsamples,Nseqs,horz_dt,horz_offset) 
            != (self.Nsamples,self.Nseqs,self.horz_dt,self.horz_offset)):
            raise TraceSetPointsChanged('prepared setpoints do not match \
                                        with waveform, run prepare_waveform')

        Npts = int(Nsamples/Nseqs)

        dat1_array = np.ndarray((Npts,),
                                'h' if comm_type else 'b',
                                wave, dat1_offset)
        # scale y values
        out_array = vert_offset + vert_gain * dat1_array

        return out_array


class TD44MXsBChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, channel_id: str) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel
            channel: The name used by the TD44MXsB (Cn)
        """
        super().__init__(parent, name)

        self.add_parameter(name='waveform',
                           parameter_class=WaveformArray,
                           channel_id=channel_id
                           )

        self.add_parameter(name='waveform_seq',
                           parameter_class=WaveformArraySeq,
                           channel_id=channel_id
                           )

        self.add_parameter(name='volt_div',
                           label='vertical sensitivity',
                           unit='V',
                           set_cmd=f'{channel_id}:VOLT_DIV {{}} V',
                           get_cmd=f'{channel_id}:VOLT_DIV?',
                           vals=vals.Numbers(min_value=0),
                           get_parser=partial(ans_parser, parser=float)
                           )

        self.add_parameter(name='offset',
                           label='vertical offset',
                           unit='V',
                           set_cmd=f'{channel_id}:OFFSET {{}} V',
                           get_cmd=f'{channel_id}:OFFSET?',
                           vals=vals.Numbers(),
                           get_parser=partial(ans_parser, parser=float)
                           )

        self.channel_id = channel_id


class TD44MXsB(VisaInstrument):
    """
    QCodeS driver for Teledyne LeCroy Wavesurfer 44Mxs-B oscilloscope
    """

    def __init__(self, name, address=None, termination='\n', **kwargs):
        """
        Args:
            name: The name of the instrument
            address: The VISA resource name of the instrument
                (e.g. "tcpip0::192.168.1.2::7230::socket")
        """
        super().__init__(name, address, **kwargs)

        # set communication format (from MAUI Remote Control Manual)
        self.visa_handle.timeout = 5000
        self.visa_handle.clear()
        self.write("COMM_HEADER OFF")
        self.write("COMM_ORDER LO")
        self.write("COMM_FORMAT OFF,BYTE,BIN")

        self.add_parameter(name='trig_mode',
                           label='trigger mode',
                           set_cmd='TRIG_MODE {}',
                           get_cmd='TRIG_MODE?',
                           vals=vals.Enum('auto', 'norm', 'single', 'stop'),
                           get_parser=ans_parser
                           )

        self.add_parameter(name='trig_delay',
                           label='trigger delay',
                           unit='s',
                           set_cmd='TRIG_DELAY {} S',
                           get_cmd='TRIG_DELAY?',
                           vals=vals.Numbers(),
                           get_parser=partial(ans_parser, parser=float)
                           )
                           
        self.add_parameter(name='horizontal_origin',
                           label='horizontal origin',
                           unit='div',
                           set_cmd="VBS 'app.Acquisition.Horizontal.HorOffsetOrigin={}'",
                           get_cmd="VBS? 'Return=app.Acquisition.Horizontal.HorOffsetOrigin'",
                           vals=vals.Numbers(0, 10),
                           get_parser=partial(ans_parser, parser=float)
                           )

        self.add_parameter(name='time_div',
                           label='horizontal sensitivity',
                           unit='s',
                           set_cmd='TIME_DIV {} S',
                           get_cmd='TIME_DIV?',
                           vals=vals.Numbers(min_value=0),
                           get_parser=partial(ans_parser, parser=float)
                           )

        self.add_parameter(name='memory_size',
                           label='number of samples',
                           set_cmd='MEMORY_SIZE {}',
                           get_cmd='MEMORY_SIZE?',
                           vals=vals.Ints(min_value=0),
                           get_parser=partial(ans_parser, parser=int)
                           )

        self.add_parameter(name='sequence',
                           label='sequence mode status',
                           set_cmd='SEQUENCE {}',
                           get_cmd='SEQUENCE?',
                           vals=vals.Enum('off', 'on'),
                           get_parser=lambda ans: ans.strip().split(',')[0]
                           )

        self.add_parameter(name='seq_segments',
                           label='number of segments in sequence',
                           set_cmd="VBS 'app.Acquisition.Horizontal.NumSegments={}'",
                           get_cmd="VBS? 'Return=app.Acquisition.Horizontal.NumSegments'",
                           vals=vals.Ints(min_value=0),
                           get_parser=ans_parser
                           )

        # command for VBS + enum types: """ VSB 'cmd = "enum" ' """, see MAUI manual
        self.add_parameter(name='trig_type',
                           label='trigger type',
                           set_cmd="""VBS 'app.Acquisition.Trigger.Type="{}" ' """,
                           get_cmd="VBS? 'Return=app.Acquisition.Trigger.Type'",
                           vals=vals.Enum("edge", "width", "qualified", "interval", "glitch",
                                          "TV", "pattern", "dropout", "runt", "slewrate"),
                           get_parser=ans_parser
                           )

        self.add_parameter(name='trig_source',
                           label='trigger source',
                           set_cmd="""VBS 'app.Acquisition.Trigger.Edge.Source="{}" ' """,
                           get_cmd="VBS? 'Return=app.Acquisition.Trigger.Edge.Source'",
                           vals=vals.Enum("C1", "C2", "C3", "C4", "Ext", "Extdivide10", "line"),
                           get_parser=ans_parser
                           )

        self.add_parameter(name='trig_level',
                           label='trigger level',
                           unit='V',
                           set_cmd='TRIG_LEVEL {} S',
                           get_cmd='TRIG_LEVEL?',
                           vals=vals.Numbers(),
                           get_parser=partial(ans_parser, parser=float)
                           )

        self.add_parameter(name='trig_slope',
                           label='trigger slope',
                           set_cmd='TRIG_SLOPE {}',
                           get_cmd='TRIG_SLOPE?',
                           vals=vals.Enum('neg', 'pos'),
                           get_parser=ans_parser
                           )
        
        for ch in range(1, 5):
            ch_name = 'C{}'.format(ch)
            channel = TD44MXsBChannel(self, ch_name, ch_name)
            self.add_submodule(ch_name, channel)

    def _read_ieee_block(self):
        "Read IEEE block"
        # IEEE block binary data is prefixed with #lnnnnnnnn
        # where l is length of n and n is the
        # length of the data
        # ex: #800002000 prefixes 2000 data bytes

        ch = self.visa_handle.read_bytes(1)

        if len(ch) == 0:
            return b''

        while ch != b'#':
            ch = self.visa_handle.read_bytes(1)

        l = int(self.visa_handle.read_bytes(1))
        if l > 0:
            num = int(self.visa_handle.read_bytes(l))
            raw_data = self.visa_handle.read_bytes(num)
        else:
            raw_data = self.visa_handle.read_bytes()

        return raw_data
