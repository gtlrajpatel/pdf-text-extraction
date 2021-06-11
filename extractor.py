import os
import PyPDF2

from tika import parser

def pdf_to_text(file):
    """
    This function uses PyPDF2 library to extract text from PDF files
    """
    # give the path of your pdf file
    file_obj = open(file, 'rb')
    filereader=PyPDF2.PdfFileReader(file_obj, strict=False)

    total_pages = filereader.numPages
    count = 0
    str = ""

    # the while loop will read each page
    # extractText() will extract the text of each page from PDF
    while count < total_pages:
        str += filereader.getPage(count).extractText() + "\n" + "**End Page**" + "\n"
        count += 1

    filename = os.path.splitext(os.path.basename(file))[0]

    # the following line will generate the output file
    with open("output/"+ filename + ".txt", "w") as f:
        f.write(str)
        f.close()

def text_extraction_using_tika(file):
    """
    This function uses Tika library to extract text from files
    """
    parsed_file = parser.from_file(file)
    data = parsed_file['content']

    filename = os.path.splitext(os.path.basename(file))[0]

    with open("output/"+ filename + "_tika.txt", "w") as f:
        f.write(data)
        f.close()


if __name__ == '__main__':
    input_files_directory = 'input'
    for filename in os.listdir(input_files_directory):
        pdf_to_text('input/'+filename)
        text_extraction_using_tika('input/'+filename)
