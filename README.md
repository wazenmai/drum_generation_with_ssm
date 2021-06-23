# Automatic Drum Beat Accompaniment Generation
This project is forked from [https://github.com/Sma1033/drum_generation_with_ssm](https://github.com/Sma1033/drum_generation_with_ssm) 

If you want to see the graph, please use jupyter notebook to run `ipynb` files.

The `requirements.txt` is for `.py` files, if you want to run the `ipynb` files, please see `original_readme.md` for more information.

## Usage
After constructing your python virtual machine...
```
pip3 install -r requirements.txt
```
Also you need to download some pretrain checkpoints and files, please read `original_readme.md` for more information.


If you want to use markov generator
```
python3 step1.py --opt="markov" --order=3 --input="input.mid" --output="output.mid"
```

If you want to use SSM generator, you need to run 5 files.
You can choose 3 different transform in `set`.
- `cqt`: Constant-Q Transform
- `hybrid_cqt`: Constant-Q Transform with higher frequencies' hop length longer than the half the filter length
- `vqt`: Variable-Q Transform
SSM generator will automatically use the data in `input_midi/**/*.mid` as input, and automatically output the song files in `output_midi`.
```
python3 step1.py --opt="ssm" --set="cqt"
python3 step2.py
python3 step3.py
python3 step4.py
python3 step5.py
```

## Results
We put some of our results in `misc/` directory for references. We recommend using DAW software to open midi file.

Feel free to generate your own drum tracks!
