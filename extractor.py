import PyPDF2
import os

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

    filename = os.path.splitext(os.path.basename(file))[0]

    # the following line will generate the output file
    with open("output/"+ filename + ".txt", "w") as f:
        f.write(str)
        f.close()

if __name__ == '__main__':
    input_files_directory = 'input'
    for filename in os.listdir(input_files_directory):
        print("\n\n ----->>>>", filename)
        pdf_to_text('input/'+filename)
