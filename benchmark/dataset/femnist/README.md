# FEMNIST

## Description

 FEMNIST has 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3600 users

## Note

We provide the [data and client mapping and train/val splitting](https://fedscale.eecs.umich.edu/dataset/femnist.tar.gz) for FEMNIST. Note that no details were kept of any of the participants age, gender, or location, and random ids were assigned to each individual. The data folder structure is as follows
```
data/
├── client_data_mapping
│   └── *.csv for client data mapping and train/test split
```
# References
The original location of this dataset is at
[https://www.nist.gov/srd/nist-special-database-19](https://www.nist.gov/srd/nist-special-database-19).

# Acknowledgement

```bibtex
@inproceedings{FEMNIST,
  author    = {Gregory Cohen and Saeed Afshar and Jonathan Tapson and Andr{\'{e}} van Schaik},
  title     = {{EMNIST:} an extension of {MNIST} to handwritten letters},
  booktitle = {arxiv.org/abs/1702.05373},
  year      = {2017},
}
```