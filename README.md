# Facer

Python package for simple face detection, alignment, and averaging using `OpenCV` and `dlib`.

# Installation

You have my 100%, money-back guarantee, that the most difficult part of using this package is installing the requirements. Once you've got `OpenCV` and `dlib` installed, the rest will be smooth sailing.

## `OpenCV`
On Mac, use [`homebrew`]() to install `OpenCV`. On Windows, I have no clue. Sorry.

```shell
brew install opencv
```

Installing `OpenCV` with `brew` did actually work for me, but it also broke my previous Python installation and all my virtual environments. So uh, good luck with that.

## `dlib`
Installing `dlib` is hopefully straightforward. You can use `pip`:

```shell
pip install dlib
```

## Python requirements
After installing `OpenCV` and `dlib`, we just need to add a few more helper libraries.

```
pip install -r requirements.txt
```
