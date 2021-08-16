import openpyxl
import numpy as np


class Load(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def write_data(self, sheet_name, column, min_row, max_row, data):
        wb = openpyxl.load_workbook(filename=self.file_name)
        sheet = wb[sheet_name]
        index = 0
        for i in range(min_row, max_row + 1, 1):
            sheet[column + str(i)].value = str(data[index])
            index += 1
        wb.save(self.file_name)
        wb.close()

    def read_data(self, sheet_name, column, min_row, max_row):
        wb = openpyxl.load_workbook(filename=self.file_name)
        sheet = wb[sheet_name]
        data = []
        for i in range(min_row, max_row + 1, 1):
            data.append(float(sheet[column + str(i)].value))
        return np.array(data)
