import PyPDF2

def pdf_to_text(file):
    # give the path of your pdf file
    file_obj = open(file, 'rb')
    filereader=PyPDF2.PdfFileReader(file_obj)

    total_pages = filereader.numPages
    count = 0
    str = ""

    # the while loop will read each page
    # extractText() will extract the text of each page from PDF
    while count < total_pages:
        str += filereader.getPage(count).extractText() + "\n" + "**End Page**" + "\n"
        count += 1

    # the following line will generate the output file
    with open("output_RP2.txt", "w") as f:
        f.write(str)
        f.close()

if __name__ == '__main__':
    file_path = 'input/RP2.pdf'
    pdf_to_text(file_path)
