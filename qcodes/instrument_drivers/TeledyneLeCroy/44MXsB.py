from functools import partial
from qcodes import Instrument, InstrumentChannel, VisaInstrument


class TD44MXsB(VisaInstrument):
    """
    QCodeS driver for Teledyne LeCroy Wavesurfer 44Mxs-B oscilloscope
    """

    def __init__(self, name, address=None, **kwargs):
        """
        Args:
            name: The name of the instrument
            address: The VISA resource name of the instrument
                (e.g. "tcpip0::192.168.1.2::7230::socket")
            password: Password for authenticating into the instrument
                (default: '123456')
            axis_names(optional): List of names to give to the individual
                channels
        """
        super().__init__(name, address, **kwargs)

        # set communication format (from MAUI Remote Control Manual)
        self.timeout = 5000
        self.clear()
        self.write("COMM_HEADER OFF")
        self.write(r"""vbs 'app.settodefaultsetup' """)
