import os

def createDirectory(pathToDirectory):
    try:
        if not os.path.exists(pathToDirectory):
            os.makedirs(pathToDirectory)
            return f'Folder has been created [{pathToDirectory}].'
        else:
            return f'Folder already exists [{pathToDirectory}].'
    except Exception as e:
        return f'Error when creating a folder. Error: {e}'