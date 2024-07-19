import time
class log:
    logLevel = 0
    logLevelIni = 0

    def __init__( self, pathToExternalFile=None ): #None if only Terminal log is required
        if pathToExternalFile != None:
            self.logExternal = logToFile(pathToExternalFile)
        else:
            self.logExternal = None

    def resetLogLevel(self):
        self.logLevel = self.logLevelIni

    def setLogLevelIni(self, logLevel):
        self.logLevelIni = logLevel
        self.resetLogLevel()

    def setLogLevel(self, logLevel):
        self.logLevel = logLevel
    
    def getTime( self ):
        currentTime = time.localtime()
        return time.strftime('%H:%M:%S', currentTime)
            
    def error( self, message ):
        if( log.logLevel<3 ):
            print(self.getTime()+'\t__ERROR__: '+message)
            if self.logExternal:
                self.logExternal.error(message)

    def warning( self, message ):
        if( log.logLevel<2 ):
            print(self.getTime()+'\t__WARNING:__ '+message)
            if self.logExternal:
                self.logExternal.warning(message)
        
    def info( self, message ):
        if( log.logLevel<1 ):
            print(self.getTime()+'\t'+message)
            if self.logExternal:
                self.logExternal.info(message)

class logToFile:
    logLevel = 0
    
    def __init__(self, filePath):
        self.path = filePath

    def getTime( self ):
        currentTime = time.localtime()
        return time.strftime('%H:%M:%S', currentTime)
            
    def error( self, message ):
        if( self.logLevel<3 ):
            file = open(self.path, "a")
            file.write(self.getTime()+'\t__ERROR__: '+message)
            file.write('\n')
            file.close()

    def warning( self, message ):
        if( self.logLevel<2 ):
            file = open(self.path, "a")
            file.write(self.getTime()+'\t__WARNING:__ '+message)
            file.write('\n')
            file.close()
        
    def info( self, message ):
        if( self.logLevel<1 ):
            file = open(self.path, "a")
            file.write(self.getTime()+'\t'+message)
            file.write('\n')
            file.close()