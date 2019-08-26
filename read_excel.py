import xlrd as xl
import os
#import numpy
directory = "C:\\Users\\dorin\\Documents\\semester_6\\projectA\\xl_trying"
#directory = "C:\Users\dorin\Documents\semester_6\projectA\xl_trying"

col = 0
out_mat = []
#out_mat = numpy.zeros((200,10000))
for filename in os.listdir(directory):
    #f = open(filename)
    wb = xl.open_workbook(str(directory))
    #wb = xl.open_workbook(str(directory)+'\\'+str(filename[:-4]))

    print(str(directory)+'\\'+str(filename))
    sheet = wb.sheet_by_index(0)
    n_rows = sheet.nrows
    n_samp = n_rows/200
    for row in n_rows:
        out_mat.append(sheet.cell_value(row, 1))
        row = row + n_samp

    col = col + 1