import dotenv # pip install python-dotenv
import os

class NNConfiguration:
    # Load FilePath
    def load_config():
        FILE = os.getenv('FILE')

        return FILE