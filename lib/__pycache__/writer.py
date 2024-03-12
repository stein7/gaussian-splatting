import numpy as np
import pandas as pd
from os import makedirs

class SummaryWriter ():
    def __init__ (self):
        self.data_list = None
        self.data      = None
        return
    def CsvWrite (self, file_path):

