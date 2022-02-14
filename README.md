# Geometric Representation Learning

## Usage
We provide classes in three modules:
* `run_test.py`: The main script for running the experiments.
* `gbp.models`: Main directory of the models.
* `gbp.datamodules`: The main directory of the data modules.

The core modules in `GBP` are meant to be as general as possible, but you will likely have to modify `GBP.modules` and `GBP.models` for your specific application, with the existing classes serving as examples.

### Data

The datasets are automatically downloaded when the test script is called. Alternatively, for the CPD task, you can run `./data/download_cath.sh` to get the CATH 4.2 processed dataset. The datasets will be stored in the following directories:

* `/data/`: CPD dataset.
* `/atom3d-data/`: PSR and LBA datasets.



### Testing


To reproduce the results for CPD task please use the following command:
```bash
python run_test.py ./models/cpd_model.pt cpd
```

To reproduce the results for PSR task please use the following command:
```bash
python run_test.py ./models/psr_model.pt psr
```

To reproduce the results for PSR task please use the following command:
```bash
python run_test.py ./models/lba_model.pt lba
```




