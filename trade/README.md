
### Environment Configuration

- install python3 with tkiner

```shell
brew install tcl-tk
brew uninstall python3  #uninstall matplotlib, scipy, numpy first
brew install python3 --with-tcl-tk
```

- reinstall virtualenv

```shell
pip3 install virtualenv
```
- activate virtualen

```shell
virtualenv env 
source /env/bin/activate
#deactivate
/bin/deactivate
```
- install packages

```shell
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
```

- Test Environment

```shell
python3 test_panda.py
python3 test_matplot.py
```




