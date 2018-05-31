# Qcodes instrument driver for ZI UHFLI
import time
import zhinst.utils
import numpy as np
from math import sqrt

import logging
from functools import partial
from typing import Callable, List, Union, cast

import qcodes as qc
from qcodes import (Instrument, VisaInstrument, ManualParameter, 
                    MultiParameter, validators as vals)
from qcodes.instrument.channel import InstrumentChannel

log = logging.getLogger(__name__)

class PollDemodSample(MultiParameter):
    """
    Multi parameter class for the ZIUHFLI instrument for polling demod samples.
    """
    
    def __init__(self, name: str, instrument: str, **kwargs) -> None:
        super().__init__(name, names=('',), shapes=((1,),), **kwargs)
        self._instrument = instrument

    def build_poll(self):
        """
        Polling setup. Call before executing poll. Sets up the MultiParameter
        subclass correctly.
        """

        log.info('Build poll.')

        nodes = self._instrument._poll_demod_list
        device = self._instrument.device

        sigunits = {'x': 'V', 'y': 'V', 'r': 'V', 'phi': 'deg'}        
        # set required/optional arguments for MultiParameter class
        names = []
        units = []
        labels = []
        numbers = []
        tosubscribe = []
        for node in nodes:
            temp = node.split('/')
            name = 'poll_demod{}_{}'.format(str(int(temp[-3])+1), temp[-1])
            label = 'Polled {} sample of demod {}'.format(temp[-1],
                        str(int(temp[-3])+1))
            names.append(name)
            labels.append(label)
            units.append(sigunits[temp[-1]])
            numbers.append(temp[-3])


        # nodes to subscribe to '/dev.../demods/n/sample'. Only necessary once
        # per demodulator. 
        tosub = set(numbers)
        for sub in tosub:
            setstr = '/{}/demods/{}/sample'\
                        .format(device, sub)
            tosubscribe.append(setstr)
    
        # add timestamps once per demodulator number
        for sub in tosub:
            name = 'poll_demod{}_timestamps'.format(str(int(sub)+1))
            label = 'Polled timestamps of demod {}'.format(str(int(sub)+1))
            unit = 's'
            names.append(name)
            labels.append(label)
            units.append(unit)

        # update required arguments
        self.names = tuple(names)
        self.units = tuple(units)
        self.labels = tuple(labels)

        self._tosubscribe = tosubscribe
        self._unique_demods = tosub

        self._instrument.poll_correctly_set_up = True

    def get(self):
        """
        Execute polling and return data of subscribed nodes.

        Returns:

        Raises:
            Value Error: If no node has been added.
            Value Error: If PollDemodSample.build_poll has not been called
                before.
        """

        tosubscribe = self._tosubscribe
        daq = self._instrument.daq
        params = self._instrument.parameters
        duration = params['poll_demod_duration'].get()
        
        if self._tosubscribe == []:
            raise ValueError('No nodes added for polling!')

        if self._instrument.poll_correctly_set_up is False:
            raise ValueError('The poll has not been correctly set up. Please\
                                run PollDemodSample.build_poll')

        # clear buffer
        daq.sync()
        # subscribe nodes
        for node in tosubscribe:
            daq.subscribe(node)
        # poll data
        datadict = daq.poll(duration, 500, 1, True)
        # unsubscribe nodes
        for node in tosubscribe:
            daq.unsubscribe(node)        

        return self._parsedata(datadict)

    def _parsedata(self, datadict: dict):
        """
        Parses the data dict returned by the daq.poll() method and matches it
        with the nodes added by the user.

        Returns:
            tuple: The recorded data in a tuple.
        """

        returndata = []
        nodes = self._instrument._poll_demod_list
        tosubscribe = self._tosubscribe
        device = self._instrument.device
        unique_demods = self._unique_demods
        clockbase = self._instrument.clockbase.get()

        # compute r and phi
        for node in tosubscribe:
            datadict[node]['r'] = np.abs(datadict[node]['x'] 
                                    + 1j * datadict[node]['y'])
            datadict[node]['phi'] = np.angle(datadict[node]['x'] 
                                    + 1j * datadict[node]['y'], deg=True)

        # get data of user selected nodes
        for node in nodes:
            path = '/'.join(node.split('/')[:-1])
            attr = node.split('/')[-1]
            data = datadict[path][attr]
            returndata.append(data)
        
        # get timestamps
        for i in unique_demods:
            path = '/{}/demods/{}/sample'.format(device, i)
            attr = 'timestamp'
            data = datadict[path][attr]/clockbase
            data = data-data[0]
            returndata.append(data)
        
        # update shapes
        shape_list = []
        for arr in returndata:
            l = (len(arr),)
            shape_list.append(l)
        self.shapes = tuple(shape_list)

        return tuple(returndata)

class ZIUHFLI(Instrument):

    def __init__(self, name: str, device_ID: str, **kwargs) -> None:
        """
        Create an instance of the instrument.
        Args:
            name (str): The internal QCoDeS name of the instrument
            device_ID (str): The device name as listed in the web server.
        """

        # derive from QCoDeS Instrument class
        super().__init__(name, **kwargs)

        # instantiate ZI api session
        self.api_level = 6
        zisession = zhinst.utils.create_api_session(device_ID, self.api_level)
        (self.daq, self.device, self.props) = zisession

        self.daq.setDebugLevel(3)

        # instantiate modules
        self.DAQmod = self.daq.dataAcquisitionModule()        

        # add instrument parameters        
        #----------------------------- Lock-In --------------------------------

        ### clockbase ###
        self.add_parameter('clockbase',
                            label = 'Clockbase of instrument',
                            unit = 'timestamps/s',
                            get_cmd = partial(self._getter_toplvl, 
                                                'clockbase', 1))
        
        ### oscillators ###
        for osc in range(1,9):
            self.add_parameter('oscillator{}_freq'.format(osc),
                                label = 'Frequency of oscillator {}'
                                .format(osc),
                                unit = 'Hz',
                                set_cmd = partial(self._setter, 'oscs',
                                             osc-1, 1, 'freq'),
                                get_cmd = partial(self._getter, 'oscs',
                                             osc-1, 1, 'freq'),
                                vals = vals.Numbers(0, 600e6))

        ### demodulators ###
        for demod in range(1,9):

            # oscillator selected for demodulator
            self.add_parameter('demod{}_oscselect'.format(demod),
                                label = 'Selected oscillator for demod {}'
                                .format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'oscselect'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'oscselect'),
                                vals = vals.Numbers(1, 8))

            # harmonic for demodulation
            self.add_parameter('demod{}_harmonic'.format(demod),
                                label = 'Harmonic of reference frequency \
                                for demod {}'.format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'harmonic'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'harmonic'),
                                vals = vals.Numbers(1, 1023))

            # demod frequency
            self.add_parameter('demod{}_freq'.format(demod),
                                label = 'Demodulation frequency \
                                for demod {}'.format(demod),
                                unit = 'Hz',
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'freq'))

            # phase shift
            self.add_parameter('demod{}_phaseshift'.format(demod),
                                label = 'Phaseshift of demod {}'
                                .format(demod),
                                unit = 'deg',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 1, 'phaseshift'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 1, 'phaseshift'),
                                vals = vals.Numbers(0, 360))

            # phase adjust
            setstring = '/{}/demods/{}/phaseadjust'.format(self.device, demod)
            self.add_function('demod{}_phaseadjust'.format(demod),
                               call_cmd = partial(self._setter, 'demods',
                                            demod-1, 0, 'phaseadjust', 1))
            del setstring

            # adc (input) select
            adcselect_mapping = {'Sig In 1': 0,
                                 'Sig In 2': 1,
                                 'Trigger 1': 2,
                                 'Trigger 2': 3,
                                 'Aux Out 1': 4,
                                 'Aux Out 2': 5,
                                 'Aux Out 3': 6,
                                 'Aux Out 4': 7,
                                 'Aux In 1': 8,
                                 'Aux In 2': 9,
                                 'Phase Demod 4': 10,
                                 'Phase Demod 8': 11}
            
            self.add_parameter('demod{}_adcselect'.format(demod),
                                label = 'Input selected for demod \
                                {}'.format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'adcselect'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'adcselect'),
                                val_mapping = adcselect_mapping,
                                vals = vals.Enum(*list(adcselect_mapping
                                                    .keys())))

            # filter order
            self.add_parameter('demod{}_order'.format(demod),
                                label = 'Filter order of demod {}'
                                .format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'order'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'order'),
                                val_mapping = adcselect_mapping,
                                vals = vals.Numbers(1, 8))

            # time constant
            self.add_parameter('demod{}_timeconstant'.format(demod),
                                label = 'Timeconstant of demod {}'
                                .format(demod),
                                unit = 's',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 1, 'timeconstant'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 1, 'timeconstant'),
                                vals = vals.Numbers(30e-9, 76))

            # bypass lowpass filter
            self.add_parameter('demod{}_bypassfilter'.format(demod),
                                label = 'Bypass lowpass filter of demod \
                                {}'.format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'bypass'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'bypass'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())
            

            # sinc filter
            self.add_parameter('demod{}_sinc'.format(demod),
                                label = 'Enable sinc filter for demod \
                                {}'.format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'sinc'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'sinc'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # enable data transfer
            self.add_parameter('demod{}_enable'.format(demod),
                                label = 'Enable data transfer for demod \
                                {}'.format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'enable'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'enable'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # set data transfer rate
            self.add_parameter('demod{}_rate'.format(demod),
                                label = 'Sample rate for demod \
                                {}'.format(demod),
                                unit = 'Sa/s',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 1, 'rate'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 1, 'rate'),
                                vals = vals.Numbers(0, 1.6e6),
                                docstring = """
                                            Note: The value inserted by the
                                            user may be approximated to the
                                            nearest value supported by the
                                            instrument.
                                            """)

            trig_mapping = {'Continuous': 0,
                            'Trigger Input 3: rising edge': 1,                
                            'Trigger Input 3: falling edge': 2,                
                            'Trigger Input 3: both edges': 3,                
                            'Trigger Input 4: rising edge': 4,                
                            'Trigger Input 3 or 4: rising edge': 5,                
                            'Trigger Input 4: falling edge': 8,                
                            'Trigger Input 3 or 4: falling edge': 10,                
                            'Trigger Input 4: both edges': 12,                
                            'Trigger Input 3 or 4: both edges': 15}

            # demod trigger
            self.add_parameter('demod{}_trigger'.format(demod),
                                label = 'trigger mode for demod \
                                {}'.format(demod),
                                unit = '',
                                set_cmd = partial(self._setter, 'demods',
                                             demod-1, 0, 'trigger'),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 0, 'trigger'),
                                val_mapping = trig_mapping,
                                vals = vals.Enum(*list(trig_mapping.keys())))
            
            # demod sample (to get single sample)
            self.add_parameter('demod{}_sample'.format(demod),
                                label = 'single sample for demod \
                                {}'.format(demod),
                                get_cmd = partial(self._getter, 'demods',
                                             demod-1, 2, 'sample'),
                                snapshot_value = False)

            # demod sample (to get single specific x, y, R, phi sample)
            for demod_par in ['x', 'y', 'r', 'phi']:
                if demod_par in ('x', 'y', 'r'):
                    unit = 'V'
                else:
                    unit = 'deg'
                self.add_parameter('demod{}_{}'.format(demod, demod_par),
                                label = 'single sample for demod \
                                {} {}'.format(demod, demod_par),
                                unit = unit,
                                get_cmd = partial(self._get_demod_sample, 
                                            demod-1, demod_par),
                                snapshot_value = False)
                                
        ### poll demod ###
        self.add_parameter('PollDemod',
                            parameter_class=PollDemodSample)
        
        self.add_parameter('poll_demod_duration',
                            parameter_class = ManualParameter,
                            initial_value = 1,
                            label = 'demod poll duration',
                            unit = 's',
                            vals = vals.Numbers(0,999))

        # list of the demod nodes to be polled ('x', 'y', 'r', 'phi')
        self._poll_demod_list = []        

        ### signal inputs ###
        for sigin in range(1,3):

            # enable input
            self.add_parameter('input{}_enable'.format(sigin),
                                label = 'Enable input {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 0, 'on'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 0, 'on'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())
            
            # input range
            self.add_parameter('input{}_range'.format(sigin),
                                label = 'Range for input {}'.format(sigin),
                                unit = 'V',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 1, 'range'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 1, 'range'),
                                vals = vals.Numbers(10e-3, 1.5))

            # input maximum
            self.add_parameter('input{}_max'.format(sigin),
                                label = 'Maximum of input {}'.format(sigin),
                                unit = 'V',
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 1, 'max'))

            # input minimum
            self.add_parameter('input{}_min'.format(sigin),
                                label = 'Minimum of input {}'.format(sigin),
                                unit = 'V',
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 1, 'min'))

            # auto range
            self.add_parameter('input{}_autorange'.format(sigin),
                                label = 'Autorange input {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 0, 'autorange'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 0, 'autorange'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # input anti aliasing filter ('off': 900MHz, 'on': 600 MHz)
            self.add_parameter('input{}_bw'.format(sigin),
                                label = 'Input anti aliasing filter of input \
                                {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 0, 'bw'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 0, 'bw'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # scaling
            self.add_parameter('input{}_scaling'.format(sigin),
                                label = 'Scaling for input {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 1, 'scaling'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 1, 'scaling'),
                                vals = vals.Numbers(0, 1e8))

            diff_mapping = {'off': 0,
                            'Inverted': 1,
                            'Input 2 - Input 1': 3,
                            'Input 1 - Input 2': 4}

            # differential input
            self.add_parameter('input{}_diff'.format(sigin),
                                label = 'Scaling for input {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 0, 'scaling'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 0, 'scaling'),
                                val_mapping = diff_mapping,
                                vals = vals.Enum(*list(diff_mapping.keys())))
            
            # 50 Ohm input impedance
            self.add_parameter('input{}_imp50'.format(sigin),
                                label = '50 Ohm input impedance for input \
                                {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 0, 'imp50'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 0, 'imp50'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # AC coupling
            self.add_parameter('input{}_ac'.format(sigin),
                                label = 'AC coupling input {}'.format(sigin),
                                unit = '',
                                set_cmd = partial(self._setter, 'sigins',
                                             sigin-1, 0, 'ac'),
                                get_cmd = partial(self._getter, 'sigins',
                                             sigin-1, 0, 'ac'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

        ### signal outputs ###
        for sigout in range(1,3):

            # switch output on or off
            self.add_parameter('signal_output{}_on'.format(sigout),
                                label = 'Turn signal output {} on and off'
                                .format(sigout),
                                unit = '',
                                set_cmd = partial(self._sigout_setter, 
                                            'sigouts', sigout-1, 0, 'on'),
                                get_cmd = partial(self._getter,
                                            'sigouts', sigout-1, 0, 'on'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # autorange
            self.add_parameter('signal_output{}_autorange'.format(sigout),
                                label = 'Autorange signal output {}'
                                .format(sigout),
                                unit = '',
                                set_cmd = partial(self._sigout_setter, 
                                            'sigouts', sigout-1, 0, 
                                            'autorange'),
                                get_cmd = partial(self._getter, 
                                            'sigouts', sigout-1, 0, 
                                            'autorange'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            # range
            self.add_parameter('signal_output{}_range'.format(sigout),
                                label = 'Range of output {}'
                                .format(sigout),
                                unit = 'V',
                                set_cmd = partial(self._sigout_setter, 
                                            'sigouts', sigout-1, 1, 'range'),
                                get_cmd = partial(self._getter, 
                                            'sigouts', sigout-1, 1, 'range'),
                                vals = vals.Enum(0.075, 0.15, 0.75, 1.5))

            # 50 Ohm or high Z output impedance (0: high Z, 1: 50 Ohm)
            self.add_parameter('signal_output{}_imp50'.format(sigout),
                                label = '50 Ohm impedance output {}'
                                .format(sigout),
                                unit = '',
                                set_cmd = partial(self._sigout_setter, 
                                            'sigouts', sigout-1, 0, 'imp50'),
                                get_cmd = partial(self._getter, 
                                            'sigouts', sigout-1, 0, 'imp50'),
                                val_mapping = {'off': 0, 'on': 1},
                                vals = vals.OnOff())

            self.add_parameter('signal_output{}_ampdef'.format(sigout),
                                get_cmd=None, set_cmd=None,
                                initial_value='Vpk',
                                label="Signal output amplitude's definition",
                                unit='V',
                                vals=vals.Enum('Vpk','Vrms', 'dBm'))

            # offset
            self.add_parameter('signal_output{}_offset'.format(sigout),
                                label = 'Offset of output {}'
                                .format(sigout),
                                unit = 'V',
                                set_cmd = partial(self._sigout_setter, 
                                            'sigouts', sigout-1, 1, 'offset'),
                                get_cmd = partial(self._getter, 
                                            'sigouts', sigout-1, 1, 'offset'),
                                vals = vals.Numbers(-1.5, 1.5))            

            # overload
            self.add_parameter('signal_output{}_overload'.format(sigout),
                                label = 'Overload of output {}'
                                .format(sigout),
                                unit = '',
                                get_cmd = partial(self._getter, 'sigouts',
                                             sigout-1, 0, 'over'))

            # output amplitudes and enables for demod channels
            for amp in range(1,9):                
                # amplitude
                self.add_parameter('signal_output{}_amp{}'
                                    .format(sigout, amp),
                                    label = 'Output {} amplitude from \
                                    demod {}'.format(sigout, amp),
                                    unit = 'Vpk',
                                    set_cmd = partial(self._sigout_setter, 
                                                'sigouts', sigout-1, 1,
                                                'amplitudes/{}'.format(amp)),
                                    get_cmd = partial(self._getter,
                                                'sigouts', sigout-1, 1,
                                                'amplitudes/{}'.format(amp)),
                                    vals = vals.Numbers(-1.5, 1.5))
                
                # enable
                self.add_parameter('signal_output{}_enable{}'
                                    .format(sigout, amp),
                                    label = 'Enable output {} amplitude from \
                                    demod {}'.format(sigout, amp),
                                    unit = '',
                                    set_cmd = partial(self._sigout_setter, 
                                                'sigouts', sigout-1, 0,
                                                'enables/{}'.format(amp)),
                                    get_cmd = partial(self._getter,
                                                'sigouts', sigout-1, 0,
                                                'enables/{}'.format(amp)),
                                    val_mapping = {'off': 0, 'on': 1},
                                    vals = vals.OnOff())

    def _setter(self, module: str, number: int, mode: int, setting: str,
                 value) -> None:
        """
        General function to set/send settings to the device.
        The module (e.g demodulator, input, output,..) number is counted in a
        zero indexed fashion.
        Args:
            module (str): The module (eg. demodulator, input, output, ..)
                to set.
            number (int): Module's index
            mode (bool): Indicating whether we are setting an int or double
            setting (str): The module's setting to set.
            value (int/double): The value to set.
        """

        setstr = '/{}/{}/{}/{}'.format(self.device, module, number, setting)

        if mode == 0:
            self.daq.setInt(setstr, value)
        if mode == 1:
            self.daq.setDouble(setstr, value)

    def _getter(self, module: str, number: int,
                mode: int, setting: str) -> Union[float, int, str, dict]:
        """
        General get function for generic parameters. Note that some parameters
        use more specialised setter/getters.
        The module (e.g demodulator, input, output,..) number is counted in a
        zero indexed fashion.
        Args:
            module (str): The module (eg. demodulator, input, output, ..)
                we want to know the value of.
            number (int): Module's index
            mode (int): Indicating whether we are asking for an int or double.
                0: Int, 1: double, 2: Sample
            setting (str): The module's setting to set.
        returns:
            inquered value
        """

        querystr = '/{}/{}/{}/{}'.format(self.device, module, number, setting)
        log.debug("getting %s", querystr)
        if mode == 0:
            value = self.daq.getInt(querystr)
        elif mode == 1:
            value = self.daq.getDouble(querystr)
        elif mode == 2:
            value = self.daq.getSample(querystr)
        else:
            raise RuntimeError("Invalid mode supplied")
        # Weird exception, samplingrate returns a string
        return value
    
    def _getter_toplvl(self, module: str, mode: int) -> Union[float, int, str, dict]:
            """
            General get function for generic parameters. Note that some parameters
            use more specialised setter/getters.
            The module (e.g demodulator, input, output,..) number is counted in a
            zero indexed fashion.

            Args:
                module (str): The top level module (eg. clockbase, ...)
                    we want to know the value of.

            Returns:
                inquered value
            """

            querystr = '/{}/{}'.format(self.device, module)
            log.debug("getting %s", querystr)
            if mode == 0:
                value = self.daq.getInt(querystr)
            elif mode == 1:
                value = self.daq.getDouble(querystr)
            else:
                raise RuntimeError("Invalid mode supplied")
            # Weird exception, samplingrate returns a string
            return value

    def _get_demod_sample(self, number: int, demod_par: str) -> float:
        
        log.debug("getting demod {} param {}".format(number, demod_par))
        
        if demod_par not in ['x', 'y', 'r', 'phi']:
            raise RuntimeError('Invalid demodulator parameter. Valid are: \
                                x, y, r, phi')

        mode = 2
        module = 'demods'
        setting = 'sample'
        datadict = cast(dict, self._getter(module, number, mode, setting))
        
        datadict['r'] = np.abs(datadict['x'] + 1j * datadict['y'])
        datadict['phi'] = np.angle(datadict['x'] + 1j * datadict['y'], deg=True)
        
        return datadict[demod_par]

    def _sigout_setter(self, module: str, number: int, mode: str, setting: str,
                 value) -> None:
        """
        Specific setter function for signal outputs since some of the 
        parameters depend on each other and need to be checked/updated.
        Args:
            module (str): The module (eg. demodulator, input, output, ..)
                to set. Here sigouts.
            number (int): Module's index
            mode (bool): Indicating whether we are setting an int or double
            setting (str): The module's setting to set.
            value (int/double): The value to set.
        """

        # convenient reference
        params = self.parameters

        def amp_valid():
            nonlocal value
            nonlocal number
            nonlocal setting
            toget = params['signal_output{}_ampdef'.format(number+1)]
            ampdef_val = toget.get()
            toget = params['signal_output{}_autorange'.format(number+1)]
            autorange_val = toget.get()

            #---------------------- MF option specific ------------------------
            # for the multi-frequency (MF) option the sum of all (enabled?)
            # output amplitudes must not exceed the range. Tricky: Setting an
            # amplitude and enabling it afterwards.
            
            # get enabled amplitudes (TODO: Is this step necessary?)
            amp_enabled = np.zeros(8, dtype=np.bool_)
            for i in range(1,9):
                toget = params['signal_output{}_enable{}'
                                .format(number+1, i)]
                amp_enabled[i] = toget.get()
            
            # get amplitudes
            amps = np.zeros(8, dtype=np.float_)
            for i in range(1,9):
                toget = params['signal_output{}_amp{}'
                                .format(number+1, i)]
                amps[i] = toget.get()

            # get amplitude which is ment to be set
            amp_selected = int(setting[-1])
            amp_enabled[amp_selected] = False

            # sum up all amplitudes except for the one to be set
            sum_amps = np.sum(amps[amp_enabled])
            
            #---------------------- end MF option -----------------------------

            if autorange_val == 'on':
                toget = params['signal_output{}_imp50'.format(number+1)]
                imp50_val = toget.get()
                imp50_dic = {'off': 1.5, 'on': 0.75}
                range_val = imp50_dic[imp50_val]

            else:
                so_range = params['signal_output{}_range'.format(number+1)].get()
                range_val = round(so_range, 3)

            amp_val_dict={'Vpk': lambda value: value,
                          'Vrms': lambda value: value*sqrt(2),
                          'dBm': lambda value: 10**((value-10)/20)
                         }

            if -range_val < amp_val_dict[ampdef_val](value) + sum_amps\
                > range_val:
                raise ValueError('Signal Output:'
                                 + ' Amplitude too high for chosen range.')
            value = amp_val_dict[ampdef_val](value)

        def offset_valid():
            nonlocal value
            nonlocal number
            range_val = params['signal_output{}_range'.format(number+1)].get()
            range_val = round(range_val, 3)
            # amp_val = params['signal_output{}_amplitude'.format(number+1)].get()
            # amp_val = round(amp_val, 3)

            # get enabled amplitudes (TODO: Is this step necessary?)
            amp_enabled = np.zeros(8, dtype=np.bool_)
            for i in range(1,9):
                toget = params['signal_output{}_enable{}'
                                .format(number+1, i)]
                amp_enabled[i] = toget.get()
            
            # get amplitudes
            amps = np.zeros(8, dtype=np.float_)
            for i in range(1,9):
                toget = params['signal_output{}_amp{}'
                                .format(number+1, i)]
                amps[i] = toget.get()

            # sum up all amplitudes
            sum_amps = np.sum(amps[amp_enabled])
            amp_val = round(sum_amps, 3)

            if -range_val< value+amp_val > range_val:
                raise ValueError('Signal Output: Offset too high for '
                                 'chosen range.')

        def range_valid():
            nonlocal value
            nonlocal number
            toget = params['signal_output{}_autorange'.format(number+1)]
            autorange_val = toget.get()
            imp50_val = params['signal_output{}_imp50'.format(number+1)].get()
            imp50_dic = {'off': [1.5, 0.15], 'on': [0.75, 0.075]}

            if autorange_val == "on":
                raise ValueError('Signal Output :'
                                ' Cannot set range as autorange is turned on.')

            if value not in imp50_dic[imp50_val]:
                raise ValueError('Signal Output: Choose a valid range:'
                                 '[0.75, 0.075] if imp50 is on, [1.5, 0.15]'
                                 ' otherwise.')

        def ampdef_valid():
            # check which amplitude definition you can use.
            # dBm is only possible with 50 Ohm imp ON
            imp50_val = params['signal_output{}_imp50'.format(number+1)].get()
            imp50_ampdef_dict = {'on': ['Vpk','Vrms', 'dBm'],
                                 'off': ['Vpk','Vrms']}
            if value not in imp50_ampdef_dict[imp50_val]:
                raise ValueError("Signal Output: Choose a valid amplitude "
                                 "definition; ['Vpk','Vrms', 'dBm'] if imp50 is"
                                 " on, ['Vpk','Vrms'] otherwise.")

        dynamic_validation = {'range': range_valid,
                              'ampdef': ampdef_valid,
                              'amplitudes/0': amp_valid,
                              'amplitudes/1': amp_valid,
                              'amplitudes/2': amp_valid,
                              'amplitudes/3': amp_valid,
                              'amplitudes/4': amp_valid,
                              'amplitudes/5': amp_valid,
                              'amplitudes/6': amp_valid,
                              'amplitudes/7': amp_valid,
                              'offset': offset_valid}

        def update_range_offset_amp():
            range_val = params['signal_output{}_range'.format(number+1)].get()
            offset_val = params['signal_output{}_offset'.format(number+1)].get()
            amp_val = params['signal_output{}_amplitude'.format(number+1)].get()
            if -range_val < offset_val + amp_val > range_val:
                #The GUI would allow higher values but it would clip the signal.
                raise ValueError('Signal Output: Amplitude and/or '
                                 'offset out of range.')

        def update_offset():
            self.parameters['signal_output{}_offset'.format(number+1)].get()

        def update_amp():
            for amp in range(1,9):
                self.parameters['signal_output{}_amp{}'
                                .format(number+1, amp)].get()

        def update_range():
            self.parameters['signal_output{}_autorange'.format(number+1)].get()
            self.parameters['signal_output{}_range'.format(number+1)].get()

        # parameters which will potentially change other parameters
        changing_param = {'imp50': [update_range_offset_amp, update_range],
                          'autorange': [update_range],
                          'range': [update_offset, update_amp],
                          'amplitudes/0': [update_range, update_amp],
                          'amplitudes/1': [update_range, update_amp],
                          'amplitudes/2': [update_range, update_amp],
                          'amplitudes/3': [update_range, update_amp],
                          'amplitudes/4': [update_range, update_amp],
                          'amplitudes/5': [update_range, update_amp],
                          'amplitudes/6': [update_range, update_amp],
                          'amplitudes/7': [update_range, update_amp],
                          'offset': [update_range]
                        }

        setstr = '/{}/{}/{}/{}'.format(self.device, module, number, setting)

        if setting in dynamic_validation:
            dynamic_validation[setting]()

        if mode == 0:
            self.daq.setInt(setstr, value)
        if mode == 1:
            self.daq.setDouble(setstr, value)
        
        if setting in changing_param:
            [f() for f in changing_param[setting]]
    
    def add_poll_demod(self, number: int, node: str) -> None:
        """
        Add a signal for polling. When polling is executed the corresponding
        data is returned.

        Args:
            number (int): Demodulator number (1-8). The same demodulator can be
                chosen several times for different nodes (e.g. 
                'x', 'y', 'r', 'phi', ...).
            node (str): The demodulator node to record 
                ('x', 'y', 'r', 'phi', 'auxin0', 'auxin1').
        
        Raises:
            ValueError: If a demodulator outside the allowed range is
              selected.
            ValueError: If an attribute not in the list of allowed nodes
              is selected.
        """

        valid_nodes = ['x', 'y', 'r', 'phi', 'auxin0', 'auxin1']

        # validation
        if number not in range(1, 9):
            raise ValueError('Can not select demodulator {}. Only demodulators\
                                1-8 are available.'.format(number))
        if node not in valid_nodes:
            raise ValueError('Can not select node: {}. Only the following \
                                attributes are available: '.format(node) +
                            ('{}, '*len(valid_nodes))
                                .format(*valid_nodes))
        module = 'demods'
        setting = 'sample'
        setstr = '/{}/{}/{}/{}/{}'\
                    .format(self.device, module, number-1, setting, node)

        if setstr not in self._poll_demod_list:
            self._poll_demod_list.append(setstr)

        self.poll_correctly_set_up = False

    def remove_poll_demod(self, number: int, node: str) -> None:    
        """
        Remove a signal from polling. If the signal has not been added before
        a warning is logged.
        
        Args:
            number (int): Demodulator number (1-8). The same demodulator can be
                chosen several times for different nodes (e.g. 
                'x', 'y', 'r', 'phi', ...).
            node (str): The demodulator node to remove 
                ('x', 'y', 'r', 'phi', 'auxin0', 'auxin1').
        """

        module = 'demods'
        setting = 'sample'
        setstr = '/{}/{}/{}/{}/{}'\
                    .format(self.device, module, number-1, setting, node)

        if setstr not in self._poll_demod_list:
            log.warning('Can not remove {} since it was not previously added'
                        .format(setstr))
        else:
            self._poll_demod_list.remove(setstr)

    def list_poll_demod(self) -> Union[float, str, list, dict]:

        return self._poll_demod_list
