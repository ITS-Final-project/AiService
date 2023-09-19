import dotenv # pip install python-dotenv
import os

class KeyConfiguration:
    # Load FilePath
    def load_config():
        PY_SECRET = os.getenv('PY_SECRET')
        US_SECRET = os.getenv('US_SECRET')
        MS_SECRET = os.getenv('MS_SECRET')

        return PY_SECRET, US_SECRET, MS_SECRET