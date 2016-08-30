#Codes by: Yu Wang, July 2016 Copyright@MOBVOI
# A class for reading config file

class ConfigReader(object):
    def __init__(self, config_file):
        self.config_file_name = config_file
        self.config_key_value = {}

    def ReadConfig(self):
        f = open(self.config_file_name, 'r')
        for line in f:
            line = line.strip()
            # empty line
            if len(line) == 0:
                continue
            # comment line
            if line[0] == '#':
                continue
            line = line.split('=')
            key = line[0].strip()
            value = line[1].strip()
            self.config_key_value[key] = value
        f.close()

    def GetKey(self, key):
        if key in self.config_key_value:
            return self.config_key_value[key]
        return None;

