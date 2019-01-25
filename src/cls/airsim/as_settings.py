import json
from os import path, getenv
from platform import system

class AS_Settings:

    def __init__(self, settings_path=None):

        if settings_path is not None:
            self.path = settings_path
        else:
            if system() == 'Windows':
                settings_path = getenv('HOMEPATH')

                self.path = path.join(settings_path, 'Documents', 'Airsim', 'settings.json')
            else:
                settings_path = getenv('HOME')

                self.path = path.join(settings_path, 'Documents', 'AirSim', 'settings.json')

        self.settings={}

        self.load()

    def load(self):
        if not path.isfile(self.path):
            self.settings = {'SeeDocsAt': "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
                             "SettingsVersion": 1.2}
        else:
            settings_file = open(self.path, mode='r')
            self.settings = json.load(settings_file)
            settings_file.close()

    def dump(self):
        settings_file = open(self.path, mode='w')
        self.settings = json.dump(self.settings, settings_file)
        settings_file.close()

    def set(self, setting, value):

        setting_subpaths = setting.split('/')
        setting = self.settings

        for i, subpath in enumerate(setting_subpaths):
            if setting is not None:
                if i == len(setting_subpaths) - 1:
                    setting[subpath] = value
                else:
                    if subpath in setting:
                        setting = setting[subpath]
                    else:
                        if value == {} or value == '':
                            break
                        else:
                            setting[subpath] = {}
                            setting = setting[subpath]


    def get(self, setting):

        setting_subpaths = setting.split('/')
        setting = self.settings

        for i, subpath in enumerate(setting_subpaths):
            if setting is not None:
                if i == len(setting_subpaths) - 1:
                    if subpath in setting:
                        setting = setting[subpath]
                    else:
                        setting = None

                else:
                    if subpath in setting:
                        setting = setting[subpath]
                    else:
                        setting = None
                        break

        if isinstance(setting, dict):
            setting = None

        return setting



    def clear(self, setting, is_dict):
        if is_dict:
            value = {}
        else:
            value = ''

        self.set(setting, value)


# Class Tests
if __name__ == '__main__':

    settings = AS_Settings()

    settings.set('Vehicles/SimpleFlight/VehicleType', 'SimpleFlight')
    tmp_setting = settings.get('Vehicles/SimpleFlight/VehicleType')
    print(tmp_setting)
    settings.clear('Vehicles', True)
    tmp_setting = settings.get('Vehicles/SimpleFlight')
    print(tmp_setting)
    tmp_setting = settings.get('Vehicles')
    print(tmp_setting)

    settings.dump()

    print('Done')