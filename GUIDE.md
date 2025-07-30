# Setup guide

## Windows

The following steps should let you set up the vocal remover on Windows:
1) Run PowerShell **in Administrator mode** (`cmd` should work too) and navigate to the root directory of the project - where this guide is
2) Install the requirements with `pip install -r .\requirements.txt`
3) Run the following commands to take care of potentially troublesome packages:
	- `pip install einops` 
		- The output may say that it was already installed, but there's no harm running it just in case
	- `pip show numpy`
		- If the version number differs from `1.23.5`, run `pip install numpy==1.23.5`
			- This may show an error (`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.`), but at the end it'll say that it was successfully installed, so you don't have to worry about that
	- `pip show librosa`
		- If the version number differs from `0.8.1`, run `pip install librosa==0.8.1`
	- `pip show torch`
		- If the version differs from `2.0.1+cu118`, run `python -m pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4) Navigate to the directory of the latest version (currently `v7.2+v8.1`) with: `cd .\snapshots\v7.2+v8.1\`
5) Download the `v7.2` and `v8.1` models from the links inside of `inference.py` - just search for `mega` and you'll find them around line 110
6) Place the downloaded model files (`model.v7.2.pth` and `model.v8.1.pth`) in the `snapshots\v7.2+v8.1` directory
7) Modify `convert.bat` inside of the same directory where you put the models to point to the folder containing the music files you want to convert. All the audio files inside that folder will be converted.
	- For example, the author's `convert.bat` file ended up being:
	```
	python inference.py --gpu 0 --input "D://Music/Amon Amarth - Twilight of the Thunder God" --output "D://Music/instrumental_output"
	```
	- Note the extra slash after the drive path: `D://`
8) Run the vocal remover with `.\convert.bat`
	- Note: It's important you run this with administrator privileges, or you may get an error about lacking permissions when trying to load files
		- Some people reported that they didn't have to run it in Administrator mode, so feel free to experiment
	- Some users have reported seeing `out of memory` errors when running the model on 6 GB graphics cards. No such errors could be seen on an 8 GB RTX 3070, but [Benjamin said](https://discord.com/channels/1143212618006405170/1143212618492936292/1158183982542897254) to edit the `cropsize` and `padding` arguments in `inference.py` (found on lines 124 and 125 for `v7.2+v8.1`) and halve the default values to `512` and `256`.
9) If everything works fine you should see it processing the tracks one by one. Whenever a track from the folder you specified finishes processing, you'll see the output in the folder you specified in convert.bat. On an RTX 3070 it takes between two and three minutes to convert a three-minute track.