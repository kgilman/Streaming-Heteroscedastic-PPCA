using Downloads: download
using BSON

# Download the data file (if needed)
URL  = "https://gitlab.com/dahong/optimally-weighted-pca-heteroscedastic-data/-/raw/master/Figure-8-9/cache/data,wave-1480.0-0.5-1620.0,k-5.bson"
LOCALPATH = basename(URL)
ispath(LOCALPATH) || download(URL, LOCALPATH)

# Load the data
# Yfull: data matrix (281 features x 10459 samples)
# vfull: variance vector (one variance per sample, averaged across features)
Yfull, vfull = BSON.load(LOCALPATH)[:ans]
