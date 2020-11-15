import logging

if __name__ == '__main__':
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    mylogger.addHandler(stream_handler)

    file_handler = logging.FileHandler('my.log')
    file_handler.setLevel(logging.INFO)
    mylogger.addHandler(file_handler)

    mylogger.info("server start!!!")
