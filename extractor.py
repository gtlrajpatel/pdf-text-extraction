import PyPDF2

# give the path of your pdf file
file_obj = open('input/RP2.pdf', 'rb')
filereader=PyPDF2.PdfFileReader(file_obj)

total_pages = filereader.numPages
count = 0
str = ""

# the while loop will read each page
# extractText() will extract the text of each page from PDF
while count < total_pages:
    str += filereader.getPage(count).extractText()
    count += 1

# the following line will generate the output file
with open("output_RP2.txt", "w") as f:
    f.write(str)
    f.close()
