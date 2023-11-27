# Puffer Dataset

## Description

Puffer data comprises different measurements — each measurement contains a different set of time-series data collected on Puffer servers, and is dumped as a CSV file. We select the data of the first week data of 2021. 

## Note

We provide the [data and client mapping and train/test splitting](https://fedscale.eecs.umich.edu/dataset/puffer.tar.gz). The data folder structure is as follows
```
waymo/
├── client_data_mapping
│   └── *.csv for client data mapping and train/test splitting
```
# References
The original location of this dataset is at
[Puffer Dataset](https://puffer.stanford.edu/data-description/).

# Acknowledgement

```bibtex
@inproceedings{puffer,
   author = {Francis Y. Yan and Hudson Ayers and et al.},
   title =  {Learning in situ: a randomized experiment in video streaming},
   booktitle =  {NSDI},
   year = {2020},
}
```