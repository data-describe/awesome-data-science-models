import subprocess
import setuptools
from distutils.command.build import build as _build


class build(_build):  # pylint: disable=invalid-name
    sub_commands = _build.sub_commands + [('CustomCommands', None)]

CUSTOM_COMMANDS = [
    
    #['apt', 'update']
    #['wget', 'https://pypi.python.org/packages/96/90/4e8328119e5fed3145c737beba63567d5557b1a20ffad453391aba95fbe4/opencv_python-3.3.0.10-cp27-cp27mu-manylinux1_x86_64.whl#md5=8c9d2f8bb89f5000142042c303779f82'],
    #['apt', '-y', 'install', 'libglib2.0-0'],
    #['apt', '-y', 'install', 'libsm6'],
    #['apt', '-y', 'install', 'libxrender-dev'],
    #['apt', '-y', 'install', 'libxext-dev'],
    ['pip', 'install', 'google-cloud-storage==1.6.0', '--ignore-installed'],
    #['pip', 'install', 'opencv_python-3.3.0.10-cp27-cp27mu-manylinux1_x86_64.whl'],
    #['pip', 'install', 'six==1.10.0'],
    #['pip', 'install', '--upgrade', 'pip'],
    #['pip', 'install', 'pandas-gbq==0.6.1', '--ignore-installed']
    #['apt-get', 'install', 'python-tk'],
    #['apt', '-y', 'install', 'python-tk'],
    #['pip', 'install', 'tulipy==0.4.0'], 
    #['pip', 'install', 'hyperopt==0.1.2'], 
    #['pip', 'install', 'xgboost==0.82'], 
    #['pip', 'install', 'pandas-gbq==0.10.0'], 
    #['pip', 'install', 'scikit-learn==0.20.3'], 
    #['pip', 'install', 'matplotlib==2.2.4'], 
    #['pip', 'install', 'google-cloud-storage==1.16.1'], 
    #['pip', 'install', 'mcfly==1.0.4'], 
    ['pip', 'install', 'lxml'],
    ['pip', 'install', 'beautifulsoup4']
]


class CustomCommands(setuptools.Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run_custom_command(self, command_list):
        #print 'Running command: %s' % command_list
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        stdout_data, _ = p.communicate()
        #print 'Command output: %s' % stdout_data
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.run_custom_command(command)


setuptools.setup(
    name='proj_dev_workshop',
    version='0.0.1',
    description='project_delivery',
    packages=setuptools.find_packages(),
    cmdclass={
        'build': build,
        'CustomCommands': CustomCommands,
    }
)
