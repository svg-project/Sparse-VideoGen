# Troubleshooting Guide for Sparse-VideoGen

This guide provides solutions for common installation and setup issues.

## Installing Git Large File Storage (Git LFS)

Git LFS is required for handling large files in the repository. If you do not have sudo, you can refer to [this guide]( https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab) or [this guide](https://github.com/git-lfs/git-lfs/issues/5955).

## GLIBCXX Version Error

If you encounter the error: "version `GLIBCXX_3.4.32' not found", solve it by installing the latest libstdcxx:

```bash
conda install -c conda-forge libstdcxx-ng
```

This will update the C++ standard library to the required version.

After installing, verify that the update was successful by checking the available GLIBCXX versions:
```bash
strings <your_conda_env>/lib/libstdc++.so.6 | grep GLIBCXX
```

If the output includes `GLIBCXX_3.4.32`, the update was successful.

