# BFRL
This is a model core implementation from the paper "Robust Multi-tab Website Fingerprinting Attacks for Flawed Traffic".
### Environment
Please first install the runtime environment in requestments.txt. 
```
pip install -r requirements.txt
```
#### PCAP Process
Run the Zeek script to process the traffic file.
```
zeek -Cr filename.pcap FeasExtract.zeek
```
### Train
Customize the parameters in the `config.yml` file and run the `python model.py` with `step = 0`
### Predict
Customize the parameters in the `config.yml` file and run the `python model.py` with `step = 1`

### Data collect
Run the `python datacollect.py` with custom parameters. 
