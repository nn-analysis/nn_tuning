import os


class Database:
    """
    Class for storing database wide properties

    When initialising the class the database folder will be created if it does not exist

    Args:
        folder (str) : The folder the database resides in
    """

    folder: str

    def __init__(self, folder: str):
        if folder[-1] == "/":
            folder = folder[:-1]
        self.folder = folder + "/"
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder, 0o755)
