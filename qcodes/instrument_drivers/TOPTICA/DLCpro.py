import logging
import re
from functools import partial
from time import sleep, time

from qcodes import validators as vals
from qcodes import Instrument, InstrumentChannel, VisaInstrument

log = logging.getLogger(__name__)


class DLCpro(VisaInstrument):
    """
    QCodeS driver for TOPTICA DLCpro laser controller
    """

    def __init__(self, name, address=None, **kwargs):
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
        # Wait for terminal to fire up and clear welcome message
        #sleep(0.1)
        #self.visa_handle.clear()


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

        # if err_check == 'ERROR':
        #     if answer.strip() == 'Wrong axis id':
        #         raise ANC300WrongAxisId
        #     elif answer.strip() == 'Wrong axis type':
        #         raise ANC300WrongAxisType
        #     else:
        #         raise ANC300GenericError(answer.strip())

        return msg