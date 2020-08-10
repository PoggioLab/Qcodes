try:
    import nidaqmx
    from nidaqmx import constants
    from nidaqmx.constants import Edge, AcquisitionType, TerminalConfiguration, AcquisitionType, RegenerationMode
except ImportError:
    raise ImportError('Could not find nidaqmx module.')

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes import ArrayParameter, MultiParameter, ManualParameter
from qcodes.utils import validators as vals

from typing import Tuple

from functools import partial
import numpy as np


ADC_pipeline_samps = {
    "PXI-6115": 2,
    }

DMA_len = {
    "PXI-6115": 32,
}

sample_len = {
    "PXI-6115": 2,
}


def dist(b, a):
    return np.sqrt((b[1]-a[1])**2+(b[0]-a[0])**2)


def trajectory(b, a, vel, dt):
    N = np.ceil(dist(b,a)/vel/dt)
    if N < 2:
        N=2
    return np.require(np.linspace(a, b, N).T, requirements=["C_CONTIGUOUS", "WRITEABLE"])


def params_retrig_AI_read(ai_task, N_samps):
    ai_instr_id = ai_task._parent._device.product_type

    N_chan = ai_task.number_of_channels.get()
    N_samps_pipe = ADC_pipeline_samps[ai_instr_id]  # PXI-6115 ADCs each have a 2-point pipeline. 2 extra samples/channel are required to empty them
    DMA_length = DMA_len[ai_instr_id] # PXI-6115 with extended memory option has 32 bytes FIFO
    sample_length = sample_len[ai_instr_id] # PXI-6115 has 12-bit DACs so 1 sample fits in 2 bytes

    DMA_samples = DMA_length/sample_length # PXI 6115 transfers data by chunks of 16 samples
    if DMA_samples % N_chan == 0: # We put N_chan samples in the FIFO at each sampling clock tick, if N_chan divides FIFO_samples,
        DMA_ticks = DMA_samples/N_chan # we need only DMA_samples/N_chan ticks to fill the FIFO
    else:
        DMA_ticks = DMA_samples # else we need DMA_samples ticks to fill N_chan times the FIFO
        
    N_samps_strand = DMA_ticks - N_samps_pipe % DMA_ticks
    N_samps_trail = DMA_ticks - N_samps % DMA_ticks
    N_ticks = N_samps + N_samps_trail + N_samps_strand + N_samps_pipe

    assert N_ticks % 1 == 0
    N_ticks = int(N_ticks)
    assert N_samps_trail % 1 == 0
    N_samps_trail = int(N_samps_trail)
    assert N_samps_strand % 1 == 0
    N_samps_strand = int(N_samps_strand)
    N_samps_lead = N_samps_pipe + N_samps_strand

    assert ((N_samps + N_samps_trail + N_samps_strand) * N_chan) % DMA_samples == ((N_ticks + N_samps_strand) * N_chan) % DMA_samples # Sanity check: same number of leftover samples in FIFO after first and subsequent triggers
    # print(f"Acquire {N_ticks} samples/channel, "
    #     f"on the first trigger read {N_samps + N_samps_trail} samples then discard the last {N_samps_trail}, "
    #     f"on the next triggers read {N_ticks} samples then discard the first {N_samps_lead} and the last {N_samps_trail}.")

    return N_ticks, (N_samps_lead, N_samps_trail)


class AIRead(MultiParameter):
    def __init__(self, name, instrument, **kwargs):
        super().__init__(name, names=('',), shapes=((1,),), **kwargs)

        self._task = instrument

    def prepare_AIRead(self, external_clock_dict = None, setpointlist=None):        
        self._ext_clock_dict = external_clock_dict

        N_chans = self._task.number_of_channels.get()

        if self._ext_clock_dict is not None:
            N_samps = self._ext_clock_dict['N_samps']
            sampling_rate_Hz = self._ext_clock_dict['sampling_rate_Hz']   
            N_ticks, (N_lead, N_trail) = params_retrig_AI_read(self._task, N_samps)
        else:
            N_samps = self._task.N_samps
            sampling_rate_Hz = self._task.sampling_rate_Hz
            N_ticks, (N_lead, N_trail) = N_samps, (0, 0)
            
        if setpointlist is None:
            setpointlist = tuple(np.arange(N_samps)/sampling_rate_Hz)
        else:
            assert len(setpointlist) == N_samps
            
        self.N_ticks = N_ticks
        self.N_lead = N_lead
        self.N_trail = N_trail
        self.N_samps = N_samps

        spname = 'Time'
        spunit = 's'

        self.setpoints = ((setpointlist,),)*N_chans
        self.setpoint_names = ((spname,),)*N_chans
        self.setpoint_units = ((spunit,),)*N_chans
        self.setpoint_labels = ((spname,),)*N_chans
        self.names = tuple(self._task._task.channel_names)
        self.units = ('V',)*N_chans
        self.labels = self.names
        self.shapes = ((N_samps,),)*N_chans

    def get_raw(self):
        if self._task._first_read:
            self._task._first_read = False            
            res_raw = np.array(self._task._task.read(self.N_samps + self.N_trail))
            if len(res_raw.shape) == 1:
                res_raw = res_raw[np.newaxis,:]
            res = tuple(res_raw[:,0: self.N_samps + self.N_trail - self.N_trail])
        else:
            res_raw = np.array(self._task._task.read(self.N_ticks))
            if len(res_raw.shape) == 1:
                res_raw = res_raw[np.newaxis,:]
            res = tuple(res_raw[:,self.N_lead: self.N_ticks - self.N_trail])
        return res


class NIDAQ_StartTrigger(InstrumentChannel):
    def __init__(self, parent: InstrumentChannel, name):
        super().__init__(parent, name)

        trigger = self.parent._task.triggers.start_trigger

        self.add_parameter(name='trig_type',
                           label=' type',
                           set_cmd=lambda x: setattr(trigger, "trig_type", x),
                           get_cmd=lambda: getattr(trigger, "trig_type"),
                           vals=vals.Enum(*constants.TriggerType)
                           )
        
        self.add_parameter(name='dig_edge_src',
                           label='start trigger digital edge source',
                           set_cmd=lambda x: setattr(trigger, "dig_edge_src", x),
                           get_cmd=lambda: getattr(trigger, "dig_edge_src"),
                           vals=vals.Strings()
                           )
        
        self.add_parameter(name='retriggerable',
                           label='start trigger retrigerable',
                           set_cmd=lambda x: setattr(trigger, "retriggerable", x),
                           get_cmd=lambda: getattr(trigger, "retriggerable"),
                           vals=vals.Bool()
                           )


class NIDAQ_ImplicitClock(InstrumentChannel):
    def __init__(self, parent: InstrumentChannel, name,
                 sample_mode=AcquisitionType.FINITE,
                 samps_per_chan=1000):
        super().__init__(parent, name)

        timing = self.parent._task.timing

        timing.cfg_implicit_timing(sample_mode=sample_mode,
                                   samps_per_chan=samps_per_chan)
        
        self.add_parameter(name='samp_quant_samp_mode',
                           label='acquisition type',
                           set_cmd=lambda x: setattr(timing, "samp_quant_samp_mode", x),
                           get_cmd=lambda: getattr(timing, "samp_quant_samp_mode"),
                           vals=vals.Enum(*AcquisitionType)
                           )
        
        self.add_parameter(name='samp_quant_samp_per_chan',
                           label='N samples',
                           unit='',
                           set_cmd=lambda x: setattr(timing, "samp_quant_samp_per_chan", x),
                           get_cmd=lambda: getattr(timing, "samp_quant_samp_per_chan"),
                           vals=vals.Ints(min_value=0)
                           )


class NIDAQ_SampleClock(InstrumentChannel):
    def __init__(self, parent: InstrumentChannel, name, rate, source="",
                 active_edge=Edge.RISING,
                 sample_mode=AcquisitionType.FINITE,
                 samps_per_chan=1000):
        super().__init__(parent, name)

        timing = self.parent._task.timing

        timing.cfg_samp_clk_timing(rate, source, active_edge, sample_mode, samps_per_chan)

        # self.add_parameter(name='type',
        #                    label='sample timing type',
        #                    set_cmd=lambda x: setattr(timing, "samp_timing_type", x),
        #                    get_cmd=lambda: getattr(timing, "samp_timing_type"),
        #                    vals=vals.Enum(*SampleTimingType)
        #                    )

        self.add_parameter(name='samp_clk_rate',
                           label='sampling rate',
                           unit='samples/s',
                           set_cmd=lambda x: setattr(timing, "samp_clk_rate", x),
                           get_cmd=lambda: getattr(timing, "samp_clk_rate"),
                           vals=vals.Numbers(min_value=0, max_value=timing.samp_clk_max_rate)
                           )
        
        self.add_parameter(name='samp_quant_samp_mode',
                           label='acquisition type',
                           set_cmd=lambda x: setattr(timing, "samp_quant_samp_mode", x),
                           get_cmd=lambda: getattr(timing, "samp_quant_samp_mode"),
                           vals=vals.Enum(*AcquisitionType)
                           )
        
        self.add_parameter(name='samp_quant_samp_per_chan',
                           label='N samples',
                           unit='',
                           set_cmd=lambda x: setattr(timing, "samp_quant_samp_per_chan", x),
                           get_cmd=lambda: getattr(timing, "samp_quant_samp_per_chan"),
                           vals=vals.Ints(min_value=0)
                           )
        
        self.add_parameter(name='samp_clk_src',
                           label='sample clock source',
                           set_cmd=lambda x: setattr(timing, "samp_clk_src", x),
                           get_cmd=lambda: getattr(timing, "samp_clk_src"),
                           vals=vals.Strings()
                           )


class NIDAQ_COFreqChannel(InstrumentChannel):
    def __init__(self, parent: InstrumentChannel, name, chanid,
                 units=constants.FrequencyUnits.HZ,
                 idle_state=constants.Level.LOW, freq=1):
        super().__init__(parent, name)

        chan = self.parent._task.co_channels.add_co_pulse_chan_freq(
            chanid, name, units=units, idle_state=idle_state, freq=freq)
             
        self._chan = chan

        self.add_parameter(name='co_count',
                           label='co count',
                           get_cmd=lambda: getattr(chan, "co_count")
                           )

        self.add_parameter(name='co_pulse_freq',
                           label='co pulse frequency',
                           unit='Hz',
                           set_cmd=lambda x: setattr(chan, "co_pulse_freq", x),
                           get_cmd=lambda: getattr(chan, "co_pulse_freq"),
                           vals=vals.Numbers(0)
                           )

        self.add_parameter(name='co_pulse_idle_state',
                           label='co idle state',
                           set_cmd=lambda x: setattr(chan, "co_pulse_idle_state", x),
                           get_cmd=lambda: getattr(chan, "co_pulse_idle_state"),
                           vals=vals.Enum(*constants.Level)
                           )

        self.add_parameter(name='co_pulse_done',
                           label='co pulse done',
                           get_cmd=lambda: getattr(chan, "co_pulse_done")
                           )


class NIDAQ_AIVoltChannel(InstrumentChannel):
    def __init__(self, parent: InstrumentChannel, name, chanid,
                 terminal_config=TerminalConfiguration.RSE,
                 min_val=-10, max_val=10):
        super().__init__(parent, name)

        chan = self.parent._task.ai_channels.add_ai_voltage_chan(
            chanid, name, terminal_config=terminal_config, min_val=min_val, max_val=max_val)
             
        self._chan = chan

        self.add_parameter(name='ai_term_cfg',
                           label='ai terminal configuration',
                           set_cmd=lambda x: setattr(chan, "ai_term_cfg", x),
                           get_cmd=lambda: getattr(chan, "ai_term_cfg"),
                           vals=vals.Enum(*constants.TerminalConfiguration)
                           )

        self.add_parameter(name='ai_coupling',
                           label='ai coupling mode',
                           set_cmd=lambda x: setattr(chan, "ai_coupling", x),
                           get_cmd=lambda: getattr(chan, "ai_coupling"),
                           vals=vals.Enum(*constants.Coupling)
                           )

        self.add_parameter(name='ai_min',
                           label='ai min value',
                           unit='V',
                           set_cmd=lambda x: setattr(chan, "ai_min", x),
                           get_cmd=lambda: getattr(chan, "ai_min"),
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='ai_max',
                           label='ai max value',
                           unit='V',
                           set_cmd=lambda x: setattr(chan, "ai_max", x),
                           get_cmd=lambda: getattr(chan, "ai_max"),
                           vals=vals.Numbers()
                           )


class NIDAQ_AOVoltChannel(InstrumentChannel):
    def __init__(self, parent: InstrumentChannel, name, chanid,
                 min_val=-10, max_val=10):
        super().__init__(parent, name)

        chan = self.parent._task.ao_channels.add_ao_voltage_chan(
            chanid, name, min_val=min_val, max_val=max_val)
             
        self._chan = chan

        self.add_parameter(name='ao_term_cfg',
                           label='ao terminal configuration',
                           set_cmd=lambda x: setattr(chan, "ao_term_cfg", x),
                           get_cmd=lambda: getattr(chan, "ao_term_cfg"),
                           vals=vals.Enum(*constants.TerminalConfiguration)
                           )

        self.add_parameter(name='ai_coupling',
                           label='ai coupling mode',
                           set_cmd=lambda x: setattr(chan, "ai_coupling", x),
                           get_cmd=lambda: getattr(chan, "ai_coupling"),
                           vals=vals.Enum(*constants.Coupling)
                           )

        self.add_parameter(name='ao_output_type',
                           label='ao output type',
                           set_cmd=lambda x: setattr(chan, "ao_output_type", x),
                           get_cmd=lambda: getattr(chan, "ao_output_type"),
                           vals=vals.Enum(*constants.UsageTypeAO)
                           )

        self.add_parameter(name='ao_idle_output_behavior',
                           label='ao idle output behavior',
                           set_cmd=lambda x: setattr(chan, "ao_idle_output_behavior", x),
                           get_cmd=lambda: getattr(chan, "ao_idle_output_behavior"),
                           vals=vals.Enum(*constants.AOIdleOutputBehavior)
                           )

        self.add_parameter(name='ao_min',
                           label='ao min value',
                           unit='V',
                           set_cmd=lambda x: setattr(chan, "ao_min", x),
                           get_cmd=lambda: getattr(chan, "ao_min"),
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='ao_max',
                           label='ao max value',
                           unit='V',
                           set_cmd=lambda x: setattr(chan, "ao_max", x),
                           get_cmd=lambda: getattr(chan, "ao_max"),
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='ao_output_impedance',
                           label='ao output impedance',
                           unit='ohm',
                           set_cmd=lambda x: setattr(chan, "ao_output_impedance", x),
                           get_cmd=lambda: getattr(chan, "ao_output_impedance"),
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='ao_voltage_current_limit',
                           label='ao voltage current limit',
                           unit='A',
                           set_cmd=lambda x: setattr(chan, "ao_voltage_current_limit", x),
                           get_cmd=lambda: getattr(chan, "ao_voltage_current_limit"),
                           vals=vals.Numbers()
                           )


class NIDAQ_Task(InstrumentChannel):
    def __init__(self, parent: Instrument, name):
        super().__init__(parent, name)

        self._name = name
        self._task = task = nidaqmx.Task(name)
        
        self.N_samps = None # clock.samp_quant_samp_per_chan.get()
        self.sampling_rate_Hz = None # clock.samp_clk_rate.get()

        self.add_submodule("start_trigger",
                           NIDAQ_StartTrigger(self, "start_trigger"))

        self.add_parameter(name='number_of_channels',
                           label='number of channels',
                           get_cmd=lambda: getattr(task, "number_of_channels")
                           )

        self.add_parameter('AIRead',
                           parameter_class=AIRead,
                           )

    def add_co_freq_channel(self, channame, chanid,
                            units=constants.FrequencyUnits.HZ,
                            idle_state=constants.Level.LOW,
                            freq=1):
        chan = NIDAQ_COFreqChannel(self, channame, chanid, units, idle_state, freq)
        self.add_submodule(channame, chan)
        self.sampling_rate_Hz = freq

    def add_ai_volt_channel(self, channame, chanid,
                            terminal_config=TerminalConfiguration.RSE,
                            min_val=-10, max_val=10):
        chan = NIDAQ_AIVoltChannel(self, channame, chanid, terminal_config, min_val, max_val)
        self.add_submodule(channame, chan)

    def add_ao_volt_channel(self, channame, chanid,
                            min_val=-10, max_val=10):
        chan = NIDAQ_AOVoltChannel(self, channame, chanid, min_val, max_val)
        self.add_submodule(channame, chan)

    def add_sample_clock(self, rate, source="",
                         active_edge=Edge.RISING,
                         sample_mode=AcquisitionType.FINITE,
                         samps_per_chan=1000):
        clockname = "clock"
        clock = NIDAQ_SampleClock(self, clockname, rate, source,
                                  active_edge, sample_mode, samps_per_chan)
        self.add_submodule(clockname, clock)
        self.N_samps = samps_per_chan
        self.sampling_rate_Hz = rate

    def add_implicit_clock(self, sample_mode=AcquisitionType.FINITE,
                           samps_per_chan=1000):
        clockname = "clock"
        clock = NIDAQ_ImplicitClock(self, clockname, sample_mode, samps_per_chan)
        self.add_submodule(clockname, clock)
        self.N_samps = samps_per_chan

    def write(self, *args, **kwargs):
        self._task.write(*args, **kwargs)

    def start(self):
        self._task.start()
        self._first_read = True

    def stop(self):
        self._task.stop()

    def close(self):
        self._parent.close_task(self._name)


class PXI_6251(Instrument):
    """
    QCoDeS driver for NI PXI 6251 DAQ card.

    Requires nidaqmx module to be installed (pip install nidaqmx).
    """

    def __init__(self, name: str, device_name: str = 'PXI-6251', **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name (str): The internal QCoDeS name of the instrument
            device_name (str): The device name from the list
                system = nidaqmx.system.System.local()
                [device.name for device in system.devices]
        """

        super().__init__(name, **kwargs)

        self._device = nidaqmx.system.device.Device(device_name)
        self._device_name = self._device.name
        self._ai_channels = self._device.ai_physical_chans.channel_names
        self._ao_channels = self._device.ao_physical_chans.channel_names
        self._ci_channels = self._device.ci_physical_chans.channel_names
        self._co_channels = self._device.co_physical_chans.channel_names
        self._di_lines = self._device.di_lines.channel_names
        self._di_ports = self._device.di_ports.channel_names
        self._do_lines = self._device.do_lines.channel_names
        self._do_ports = self._device.do_ports.channel_names
        self._terminals = self._device.terminals
            
    def get_idn(self):
        """
        Returns:
            A dict containing vendor, model, serial, and firmware.
        """

        idn_dict ={
            'vendor': 'NI',
            'model': self._device.product_type,
            'serial': str(self._device.product_num),
            'firmware': ''
        }

        return idn_dict

    def add_task(self, taskname):
        task = NIDAQ_Task(self, taskname)
        self.add_submodule(taskname, task)

    def close_task(self, taskname):
        self.submodules[taskname]._task.close()
        del self.submodules[taskname]



class MetaXYScannerDriver(Instrument):
    """
    QCoDeS meta driver creating an XY scanner driver using 2 analog outputs from an NI PXI 6251 DAQ card.

    Requires nidaqmx module to be installed (pip install nidaqmx).
    """

    def __init__(self, name: str, daq_card_name: str, xy_channel_names: Tuple[str, str],
                 clock_rate:float, scan_speed_V_per_s:float, **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name (str): The internal QCoDeS name of the instrument
            daq_card_name (str): The device name from the list
                system = nidaqmx.system.System.local()
                [device.name for device in system.devices]
        """

        super().__init__(name, **kwargs)

        self._daq_card = PXI_6251(daq_card_name, **kwargs)

        self._daq_card.add_task("ao_task")

        self._daq_card.ao_task.add_ao_volt_channel('X', xy_channel_names[0])
        self._daq_card.ao_task.add_ao_volt_channel('Y', xy_channel_names[1])

        self._daq_card.ao_task._task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        self._daq_card.ao_task.add_sample_clock(clock_rate, sample_mode=AcquisitionType.FINITE)

        self._daq_card.ao_task._task.register_done_event(self.task_done_callback)

        self._pos = (None, None)
        self._target_pos = (None, None)

        self.add_parameter(name='x_min',
                           label='x min value',
                           unit='V',
                           set_cmd=self._daq_card.ao_task.X.ao_min.set,
                           get_cmd=self._daq_card.ao_task.X.ao_min.get,
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='x_max',
                           label='x max value',
                           unit='V',
                           set_cmd=self._daq_card.ao_task.X.ao_max.set,
                           get_cmd=self._daq_card.ao_task.X.ao_max.get,
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='y_min',
                           label='y min value',
                           unit='V',
                           set_cmd=self._daq_card.ao_task.Y.ao_min.set,
                           get_cmd=self._daq_card.ao_task.Y.ao_min.get,
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='y_max',
                           label='y max value',
                           unit='V',
                           set_cmd=self._daq_card.ao_task.Y.ao_max.set,
                           get_cmd=self._daq_card.ao_task.Y.ao_max.get,
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='scan_speed',
                           parameter_class=ManualParameter,
                           initial_value=scan_speed_V_per_s,
                           label='scanning speed',
                           unit='V_per_s',
                           vals=vals.MultiType(vals.Numbers(), vals.Enum(None))
                           )

        self.add_parameter(name='samp_rate',
                           label='sampling rate',
                           unit='samples/s',
                           set_cmd=self._daq_card.ao_task.clock.samp_clk_rate.set,
                           get_cmd=self._daq_card.ao_task.clock.samp_clk_rate.get,
                           vals=vals.Numbers(min_value=0, max_value=self._daq_card.ao_task._task.timing.samp_clk_max_rate)
                           )
        
        self.add_parameter(name='N_samp',
                           label='N samples',
                           unit='',
                           set_cmd=self._daq_card.ao_task.clock.samp_quant_samp_per_chan.set,
                           get_cmd=self._daq_card.ao_task.clock.samp_quant_samp_per_chan.get,
                           vals=vals.Ints(min_value=0)
                           )

    def go_to(self, pos: Tuple[float, float]):
        self.N_samp.set(2)
        self._daq_card.ao_task.write([[pos[0],]*2,[pos[1],]*2], auto_start=False)   # weird behavior of daq card with List instead of List[List] input to write
        self._daq_card.ao_task.start()
        self._target_pos = pos      

    def go_to_smooth(self, pos: Tuple[float, float]):
        if self.scan_speed() is None or self._pos == (None, None):
            self.go_to(pos)
        else:
            traj = trajectory(pos, self._pos, self.scan_speed(), 1/self.samp_rate())
            self.N_samp.set(traj.shape[1])
            self._daq_card.ao_task.write(traj, auto_start=False)
            self._daq_card.ao_task.start()
            self._target_pos = pos

    def scan_init(self, Vx_list, Vy_list):
        if self._pos == (None, None):
            self.go_to((min(Vx_list), min(Vy_list)))
        else:
            self.go_to_smooth((min(Vx_list), min(Vy_list)))
        while not self.is_done_moving():
            pass

        self.go_to_smooth((max(Vx_list), min(Vy_list)))
        while not self.is_done_moving():
            pass

        self.go_to_smooth((max(Vx_list), max(Vy_list)))
        while not self.is_done_moving():
            pass

        self.go_to_smooth((min(Vx_list), max(Vy_list)))
        while not self.is_done_moving():
            pass

        self.go_to_smooth((min(Vx_list), min(Vy_list)))
        while not self.is_done_moving():
            pass

    def is_done_moving(self):
        return self._daq_card.ao_task._task.is_task_done() and self._pos == self._target_pos

    def task_done_callback(self, task_handle, status, callback_data):
        self._daq_card.ao_task.stop()
        self._pos = self._target_pos
        return 0
        
    def stop(self):
        return self._daq_card.ao_task.stop()
    


