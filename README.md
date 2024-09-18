# tortoise_tts_api
Intentions: An easy library to use tortoise for inference and training

## Usage
You'll need to install the submodules required for this repo at their specific hashes, and then install the repo itself.

Let's say you have a project that you want to use the tortoise_tts_api in and already have a venv setup.

Clone the repo:
```
https://github.com/JarodMica/tortoise_tts_api.git
```

Activate your venv, cd into the cloned repo, then initalize github modules:
```
git submodule init
git submodule update --remote
```

This will clone these repos at the hash we want them at, then you can install each one of them with pip:
```
pip install modules\tortoise_tts
pip install modules\dlas
```

Now you can instal the repo itself and it'll work fine (hopefully):
```
pip install .
```

### Why not just include a direct download to the github link at that hash?
Then I have to change the installation instructions when I update what version of tortoise_tts or dlas I want to use, purely personal preference -_-