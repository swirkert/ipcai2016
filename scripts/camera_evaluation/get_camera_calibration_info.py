from xml.sax import make_parser, handler
import numpy as np
import pandas as pd


class CameraCalibration(handler.ContentHandler):
    def __init__(self):
        self.current_peak = 0.
        self.current_response = 0.
        self.all_bands = []
        self.w_start = 0.
        self.w_end = 0.
        self.w_step = 0.
        self.bandpass_filter = 0.
        self.current_content = ""

    def characters(self, content):
        self.current_content += content.strip()

    def startElement(self, name, attrs):
        self.current_content = ""

    def endElement(self, name):
        if name == "wavelength_nm":
            self.current_peak = float(self.current_content)
        elif name == "data":
            self.current_response = np.fromstring(self.current_content, sep=",")
        elif name == "response_composition":
            self.all_bands.append((self.current_peak, self.current_response))
        elif name == "analysis_range_start_nm":
            self.w_start = float(self.current_content)
        elif name == "analysis_range_end_nm":
            self.w_end = float(self.current_content)
        elif name == "analysis_resolution_nm":
            self.w_step = float(self.current_content)
        elif name == "rejection_filter":
            self.bandpass_filter = self.current_response

    def get_bands(self):
        sorted_by_first = sorted(self.all_bands, key=lambda tup: tup[0])
        only_bands = map(lambda m: np.reshape(m[1], (1, len(m[1]))), sorted_by_first)
        # stack and correct for bandpass filter:
        return np.concatenate(only_bands) * self.bandpass_filter

    def get_wavelengths(self):
        return np.arange(self.w_start, self.w_end + self.w_step, self.w_step) * 10**-9


def get_camera_calibration_info_df(filename):
    parser = make_parser()
    cc = CameraCalibration()
    parser.setContentHandler(cc)
    parser.parse(filename)
    df = pd.DataFrame(data=cc.get_bands(), columns=cc.get_wavelengths())
    return df


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    get_camera_calibration_info_df("/media/wirkert/data/Data/2016_09_08_Ximea/Ximea_software/xiSpec-calibration-data/CMV2K-SSM4x4-470_620-9.2.4.11.xml")

