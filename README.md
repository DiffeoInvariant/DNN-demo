# CRIKit Discuss

This repository is a casual discussion forum for ideas and examples
related to the constitutive relation inference toolkit.

### Note on data files

If using this repository to share data files, please put them in a
branch under refs/data/ so everyone doesn't have to transfer them and
they don't need to live forever in the repository.

```
git checkout -b jed-example-data
# add file, commit
git push -u origin jed-example-data:refs/data/jed-example-data
```

Anyone can examine that data from another repository using the
following.

```
git fetch origin refs/data/jed-example-data
git checkout FETCH_HEAD
```
