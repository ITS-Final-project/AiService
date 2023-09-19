import datetime
import json
import enum
import os

class LogFileHandler:
    __instace = None

    @staticmethod
    def get_instance():
        if LogFileHandler.__instace == None:
            LogFileHandler()
        return LogFileHandler.__instace
    
    def __init__(self):
        if LogFileHandler.__instace != None:
            raise Exception("This class is a singleton!")
        else:
            LogFileHandler.__instace = self
        
    def get_logs(self):
        """
            Gets list of all log file names from the log directory.

            Returns:
                list: List of all log file names.
            
            Raises:
                Exception: If the log directory does not exist.
        """

        log_dir = Logger.get_instance().get_directory()
        logs = []

        for file in os.listdir(log_dir):
            if file.endswith(".json"):
                logs.append(file)
        
        return logs

    def get_log(self, file_name):
        """
            Gets the log file with the given name.

            Returns:
                dict: The log file.

            Raises:
                Exception: If the log file does not exist.
        """

        log_dir = Logger.get_instance().get_directory()
        log_file = f'{log_dir}/{file_name}'

        if not os.path.exists(log_file):
            raise Exception("Log file does not exist!")
        
        with open(log_file) as f:
            data = json.load(f)

        return data

class Logger:
    __instance = None
    __log_file = None
    __open = False

    __file_directory = 'logs'

    class LogLevel(enum.Enum):
        INFO = 0
        WARNING = 1
        ERROR = 2

    def get_directory(self):
        """
        Gets the directory where the log files are stored.

        Returns:
            str: The directory where the log files are stored.
        """
        return self.__file_directory
    
    def get_log_file(self):
        """
        Gets the current log file.

        Returns:
            str: The current log file.
        """
        return self.__log_file

    @staticmethod
    def get_instance():
        if Logger.__instance == None:
            Logger()
        return Logger.__instance
    
    def open(self):
        """
        Opens the logger. If the logger is already opened, nothing happens.

        Raises:
            Exception: If the logger is already opened.
        
        Returns:
            None
        """
        if self.__open:
            return
        
        print("Opening logger")
        opener = [{
            'timestamp': datetime.datetime.now().timestamp(),
            'level': 'INFO',
            'message': 'Logger opened',
            'origin': "Logger",
            'action': 'Init'
        }]

        if not os.path.exists(self.__file_directory):
            os.makedirs(self.__file_directory)
        
        self.__log_file = f'{self.__file_directory}/log_{datetime.datetime.now().timestamp()}.json'
        
        with open(self.__log_file, 'w+') as f:
            f.write(json.dumps(opener))

        self.__open = True

    def close(self):
        """
        Closes the logger. If the logger is already closed, nothing happens.

        Raises:
            Exception: If the logger is already closed.
        
        Returns:
            None
        """
        if not self.__open:
            return
        
        self.info("Logger closed")

        self.__open = False

    def info(self, message, **kwargs):
        """
        Logs an info message to the log file.

        Args:
            message (str): The message to log.
            **kwargs: Additional key-value pairs to log.

        Raises:
            Exception: If the logger is not opened.

        Returns:
            None
        """
        self.__log(message, Logger.LogLevel.INFO, **kwargs)

    def warning(self, message, **kwargs):
        """
        Logs a warning message to the log file.

        Args:
            message (str): The message to log.
            **kwargs: Additional key-value pairs to log.

        Raises:
            Exception: If the logger is not opened.

        Returns:
            None
        """
        self.__log(message, Logger.LogLevel.WARNING, **kwargs)

    def error(self, message, **kwargs):
        """
        Logs an error message to the log file.

        Args:
            message (str): The message to log.
            **kwargs: Additional key-value pairs to log.
        
        Raises:
            Exception: If the logger is not opened.
        
        Returns:
            None
        """
        self.__log(message, Logger.LogLevel.ERROR, **kwargs)

    def __log(self, message, level=LogLevel.INFO, **kwargs):
        """
        Logs a message to the log file.

        Args:
            message (str): The message to log.
            level (LogLevel): The log level of the message.
            **kwargs: Additional key-value pairs to log.
        
        Raises:
            Exception: If the logger is not opened.
        
        Returns:
            None
        """
        if not self.__open:
            raise Exception("Logger is not opened!")
        
        log = {
            'timestamp': datetime.datetime.now().timestamp(),
            'level': level.name,
            'message': message
        }

        print("Log file: ", self.__log_file)

        for key, value in kwargs.items():
            log[key] = value

        with open(self.__log_file) as f:
            f.seek(0)
            data = json.load(f)
            data.append(log)

        with open(self.__log_file, 'w') as f:  
            json.dump(data, f)

        
    def __init__(self):
        print("Initializing logger")
        if Logger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self

        self.open()
    

        

    