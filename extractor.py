import os
import PyPDF2

from tika import parser


def pdf_to_text(file):
    """
    This function uses PyPDF2 library to extract text from PDF files
    """
    # give the path of your pdf file
    file_obj = open(file, 'rb')
    filereader = PyPDF2.PdfFileReader(file_obj, strict=False)

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
    with open("output/" + filename + ".txt", "w") as f:
        f.write(str)
        f.close()


def text_extraction_using_tika(input_directory, output_directory):
    """
    This function uses Tika library to extract text from PDF, PPT, HTML & DOC files
    """
    pdf_count, doc_count, ppt_count, html_count = 0, 0, 0, 0
    for filename in os.listdir(input_directory):
        parsed_file = parser.from_file(input_directory + '/' + filename)
        data = parsed_file['content']

        filename, ext = os.path.splitext(os.path.basename(filename))

        try:
            with open(output_directory + '/' + filename + "_using_tika.txt", "w") as f:
                f.write(data.strip())
            if ext == '.pdf':
                pdf_count += 1
            if ext == '.docx' or ext == '.doc':
                doc_count += 1
            if ext == '.pptx' or ext == '.ppt':
                ppt_count += 1
            if ext == '.html':
                html_count += 1
        except Exception as e:
            print("\n\n Writing to file %s failed due to: %s", filename, e)

    print("\n\n> Processed PDF Files: %s \n> Processed DOC Files: %s \n> "
          "Processed PPT Files: %s \n> Processed HTML Files: %s \n" % (pdf_count, doc_count, ppt_count, html_count))


def xml_extraction_using_tika(input_directory, output_directory):
    """
    This function uses Tika library to extract xml from PDF files
    """
    pdf_count = 0
    for filename in os.listdir(input_directory):
        parsed_file = parser.from_file(input_directory + '/' + filename, xmlContent=True)
        data = parsed_file['content']

        filename, ext = os.path.splitext(os.path.basename(filename))

        try:
            with open(output_directory + '/' + filename + "_xml_using_tika.txt", "w") as f:
                f.write(data.strip())
            if ext == '.pdf':
                pdf_count += 1
        except Exception as e:
            print("\n\n Writing to file %s failed due to: %s", filename, e)

    print("\n\n> Processed PDF Files: %s" % pdf_count)


if __name__ == '__main__':
    print("\n\n\n *** Text Extraction Program *** \n\n")
    print("Please select the processing method:\n"
          "1. Extraction using TIKA for non xml\n"
          "2. Extraction using TIKA for xml\n")
    user_choice = input("\nPlease Enter your choice (1 or 2):")
    input_file_directory = input("> Enter the input directory path:")
    output_file_directory = input("> Enter the output directory path:")

    if user_choice == '1':
        print("\n\n TIKA for non xml in progress...")
        text_extraction_using_tika(input_file_directory, output_file_directory)
    elif user_choice == '2':
        print("\n\n TIKA for xml in progress...")
        xml_extraction_using_tika(input_file_directory, output_file_directory)
    else:
        print("Invalid Choice")
